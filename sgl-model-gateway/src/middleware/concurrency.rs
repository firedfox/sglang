//! Request admission ownership, bounded waiting, and streaming lifetime.
//!
//! Capacity is measured in HTTP requests, not generated sequences or KV tokens.
//! A lease follows a request through handler execution and response-body polling;
//! releasing it does not imply that a remote engine has acknowledged cancellation.

use std::{
    future::Future,
    pin::Pin,
    sync::{atomic::AtomicU64, Arc},
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use http_body::{Body as _, Frame};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::debug;

use crate::core::token_bucket::TokenBucket;

/// Owns already-acquired capacity, including before response headers exist.
pub(super) struct TokenLease {
    bucket: Arc<TokenBucket>,
    tokens: f64,
}

impl TokenLease {
    fn new(bucket: Arc<TokenBucket>, tokens: f64) -> Self {
        Self { bucket, tokens }
    }

    pub(super) async fn try_acquire(bucket: Arc<TokenBucket>, tokens: f64) -> Result<Self, ()> {
        bucket.try_acquire(tokens).await?;
        Ok(Self::new(bucket, tokens))
    }
}

impl Drop for TokenLease {
    fn drop(&mut self) {
        self.bucket.return_tokens_sync(self.tokens);
    }
}

/// Holds admission capacity until the body completes, errors, or is dropped.
pub struct TokenGuardBody {
    inner: Body,
    lease: Option<TokenLease>,
}

impl TokenGuardBody {
    /// Wrap a body after the caller has acquired `tokens` from `token_bucket`.
    pub fn new(inner: Body, token_bucket: Arc<TokenBucket>, tokens: f64) -> Self {
        Self::from_lease(inner, TokenLease::new(token_bucket, tokens))
    }

    fn from_lease(inner: Body, lease: TokenLease) -> Self {
        let lease = if inner.is_end_stream() {
            // Empty responses can complete without ever polling a body frame.
            drop(lease);
            None
        } else {
            Some(lease)
        };
        Self { inner, lease }
    }
}

impl http_body::Body for TokenGuardBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let frame = Pin::new(&mut this.inner).poll_frame(cx);
        if matches!(frame, Poll::Ready(None) | Poll::Ready(Some(Err(_))))
            || (frame.is_ready() && this.inner.is_end_stream())
        {
            // Release on terminal polling as well as Drop. Keeping a completed
            // Response object alive must not retain admission capacity.
            this.lease.take();
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// The lease is owned before polling the handler, so cancellation before headers
/// and cancellation while streaming have the same exactly-once release behavior.
pub(super) async fn guard_response(
    lease: TokenLease,
    response: impl Future<Output = Response>,
) -> Response {
    let (parts, body) = response.await.into_parts();
    Response::from_parts(parts, Body::new(TokenGuardBody::from_lease(body, lease)))
}

/// An entry awaiting admission. The lease, once granted, travels through the
/// channel, so a cancelled or abandoned receiver cannot leak acquired capacity.
pub struct QueuedRequest {
    queued_at: Instant,
    permit_tx: oneshot::Sender<Result<TokenLease, StatusCode>>,
}

/// Queue metrics for monitoring.
#[derive(Debug, Default)]
pub struct QueueMetrics {
    pub total_queued: AtomicU64,
    pub current_queued: AtomicU64,
    pub total_timeout: AtomicU64,
    pub total_rejected: AtomicU64,
}

/// The semaphore bounds all live waiting requests, including the entry removed
/// from the channel by the processor. The channel also bounds cancelled entries
/// that have not yet been drained, so cancellation cannot create an unbounded
/// backlog of stale entries.
#[derive(Clone)]
pub struct ConcurrencyQueue {
    sender: mpsc::Sender<QueuedRequest>,
    waiting_slots: Arc<Semaphore>,
}

impl ConcurrencyQueue {
    pub(super) async fn acquire(&self) -> Result<TokenLease, StatusCode> {
        // Keep the slot in the request future: dropping that future immediately
        // releases the slot even when its channel entry has not been processed.
        let _waiting_slot = self
            .waiting_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;
        let (permit_tx, permit_rx) = oneshot::channel();
        self.sender
            .try_send(QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
            })
            .map_err(|err| match err {
                mpsc::error::TrySendError::Full(_) => StatusCode::TOO_MANY_REQUESTS,
                mpsc::error::TrySendError::Closed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            })?;
        permit_rx
            .await
            .unwrap_or(Err(StatusCode::INTERNAL_SERVER_ERROR))
    }
}

/// A single processor waits on queued entries without spawning hidden waiters.
pub struct QueueProcessor {
    token_bucket: Arc<TokenBucket>,
    queue_rx: mpsc::Receiver<QueuedRequest>,
    queue_timeout: Duration,
}

impl QueueProcessor {
    pub fn new(
        token_bucket: Arc<TokenBucket>,
        queue_rx: mpsc::Receiver<QueuedRequest>,
        queue_timeout: Duration,
    ) -> Self {
        Self {
            token_bucket,
            queue_rx,
            queue_timeout,
        }
    }

    pub async fn run(mut self) {
        while let Some(mut queued) = self.queue_rx.recv().await {
            if queued.permit_tx.is_closed() {
                continue;
            }
            let Some(remaining) = self.queue_timeout.checked_sub(queued.queued_at.elapsed()) else {
                let _ = queued.permit_tx.send(Err(StatusCode::REQUEST_TIMEOUT));
                continue;
            };

            tokio::select! {
                biased;
                // Do not let a cancelled head wait for capacity or its deadline.
                _ = queued.permit_tx.closed() => {},
                result = self.token_bucket.acquire_timeout(1.0, remaining) => {
                    let permit = result
                        .map(|()| TokenLease::new(self.token_bucket.clone(), 1.0))
                        .map_err(|_| StatusCode::REQUEST_TIMEOUT);
                    // A send failure drops the lease. If the receiver disappears
                    // after a successful send, the channel drops the lease instead.
                    let _ = queued.permit_tx.send(permit);
                }
            }
        }
        debug!("Concurrency queue processor stopped");
    }
}

pub struct ConcurrencyLimiter {
    pub queue_tx: Option<ConcurrencyQueue>,
}

impl ConcurrencyLimiter {
    pub fn new(
        token_bucket: Option<Arc<TokenBucket>>,
        queue_size: usize,
        queue_timeout: Duration,
    ) -> (Self, Option<QueueProcessor>) {
        match (token_bucket, queue_size) {
            (Some(bucket), size) if size > 0 => {
                let (sender, receiver) = mpsc::channel(size);
                let queue = ConcurrencyQueue {
                    sender,
                    waiting_slots: Arc::new(Semaphore::new(size)),
                };
                (
                    Self {
                        queue_tx: Some(queue),
                    },
                    Some(QueueProcessor::new(bucket, receiver, queue_timeout)),
                )
            }
            _ => (Self { queue_tx: None }, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use futures_util::{poll, stream};
    use http_body_util::BodyExt;

    use super::*;

    fn queue(
        bucket: Arc<TokenBucket>,
        capacity: usize,
        timeout: Duration,
    ) -> (ConcurrencyQueue, QueueProcessor) {
        let (limiter, processor) = ConcurrencyLimiter::new(Some(bucket), capacity, timeout);
        (limiter.queue_tx.unwrap(), processor.unwrap())
    }

    async fn wait_for_processor_to_dequeue(queue: &ConcurrencyQueue) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while queue.sender.capacity() != queue.sender.max_capacity() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("queue processor did not receive the entry");
    }

    #[tokio::test]
    async fn cancellation_before_response_headers_returns_capacity() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let lease = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let (started_tx, started_rx) = oneshot::channel();
        let request = tokio::spawn(guard_response(lease, async move {
            started_tx.send(()).unwrap();
            std::future::pending::<Response>().await
        }));
        started_rx.await.unwrap();
        assert!(bucket.try_acquire(1.0).await.is_err());
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn pending_stream_holds_capacity_until_drop() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let lease = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let mut response = guard_response(lease, async {
            Response::new(Body::from_stream(stream::pending::<
                Result<Bytes, std::io::Error>,
            >()))
        })
        .await;
        assert!(poll!(response.body_mut().frame()).is_pending());
        assert!(bucket.try_acquire(1.0).await.is_err());
        drop(response);
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn final_frame_returns_capacity_without_waiting_for_body_drop() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let lease = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let mut response =
            guard_response(lease, async { Response::new(Body::from("final")) }).await;
        let frame = response.body_mut().frame().await.unwrap().unwrap();
        assert_eq!(frame.into_data().unwrap(), Bytes::from_static(b"final"));
        assert!(bucket.try_acquire(1.0).await.is_ok());
        // A new request owns the returned token. Dropping the old body must not
        // return it a second time and admit another request over capacity.
        drop(response);
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn stream_eof_returns_capacity_without_waiting_for_body_drop() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let lease = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let mut response = guard_response(lease, async {
            Response::new(Body::from_stream(stream::empty::<
                Result<Bytes, std::io::Error>,
            >()))
        })
        .await;
        assert!(response.body_mut().frame().await.is_none());
        assert!(bucket.try_acquire(1.0).await.is_ok());
        drop(response);
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn stream_error_returns_capacity_exactly_once() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let lease = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let mut response = guard_response(lease, async {
            Response::new(Body::from_stream(stream::iter([Err::<Bytes, _>(
                std::io::Error::other("upstream disconnected"),
            )])))
        })
        .await;
        assert!(response.body_mut().frame().await.unwrap().is_err());
        assert!(bucket.try_acquire(1.0).await.is_ok());
        drop(response);
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn empty_response_does_not_need_a_body_poll_to_return_capacity() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let lease = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let response = guard_response(lease, async { Response::new(Body::empty()) }).await;
        assert!(bucket.try_acquire(1.0).await.is_ok());
        drop(response);
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn queue_capacity_includes_the_waiter_removed_from_the_channel() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let active = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let (queue, processor) = queue(bucket, 1, Duration::from_secs(5));
        let mut first = Box::pin(queue.acquire());
        assert!(poll!(first.as_mut()).is_pending());
        let worker = tokio::spawn(processor.run());
        wait_for_processor_to_dequeue(&queue).await;

        assert!(matches!(
            queue.acquire().await,
            Err(StatusCode::TOO_MANY_REQUESTS)
        ));
        drop(active);
        let lease = first.await.unwrap();
        assert_eq!(queue.waiting_slots.available_permits(), 1);
        drop(lease);
        drop(queue);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn cancelled_head_releases_waiting_slot_without_waiting_for_capacity() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let active = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let (queue, processor) = queue(bucket.clone(), 1, Duration::from_secs(5));
        let mut first = Box::pin(queue.acquire());
        assert!(poll!(first.as_mut()).is_pending());
        let worker = tokio::spawn(processor.run());
        wait_for_processor_to_dequeue(&queue).await;
        drop(first);
        assert_eq!(queue.waiting_slots.available_permits(), 1);

        let mut replacement = Box::pin(queue.acquire());
        assert!(poll!(replacement.as_mut()).is_pending());
        wait_for_processor_to_dequeue(&queue).await;
        assert!(bucket.try_acquire(1.0).await.is_err());
        drop(active);
        let lease = replacement.await.unwrap();
        drop(lease);
        drop(queue);
        worker.await.unwrap();
        assert!(bucket.try_acquire(1.0).await.is_ok());
    }

    #[tokio::test]
    async fn cancelled_queued_receiver_does_not_consume_capacity() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let (sender, receiver) = mpsc::channel(1);
        let (permit_tx, permit_rx) = oneshot::channel();
        drop(permit_rx);
        assert!(sender
            .try_send(QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
            })
            .is_ok());
        drop(sender);
        QueueProcessor::new(bucket.clone(), receiver, Duration::from_secs(1))
            .run()
            .await;
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn unread_grant_returns_capacity_when_receiver_is_dropped() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let (sender, receiver) = mpsc::channel(1);
        let (permit_tx, permit_rx) = oneshot::channel();
        assert!(sender
            .try_send(QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
            })
            .is_ok());
        drop(sender);
        QueueProcessor::new(bucket.clone(), receiver, Duration::from_secs(1))
            .run()
            .await;
        assert!(bucket.try_acquire(1.0).await.is_err());
        drop(permit_rx);
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn queue_timeout_releases_slot_without_creating_capacity() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let active = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let (queue, processor) = queue(bucket.clone(), 1, Duration::from_millis(10));
        let worker = tokio::spawn(processor.run());
        assert!(matches!(
            queue.acquire().await,
            Err(StatusCode::REQUEST_TIMEOUT)
        ));
        assert_eq!(queue.waiting_slots.available_permits(), 1);
        assert!(bucket.try_acquire(1.0).await.is_err());
        drop(active);
        assert!(bucket.try_acquire(1.0).await.is_ok());
        drop(queue);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn stopped_processor_releases_waiters_and_slots() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let _active = TokenLease::try_acquire(bucket.clone(), 1.0).await.unwrap();
        let (queue, processor) = queue(bucket, 1, Duration::from_secs(5));
        let mut waiting = Box::pin(queue.acquire());
        assert!(poll!(waiting.as_mut()).is_pending());
        let worker = tokio::spawn(processor.run());
        wait_for_processor_to_dequeue(&queue).await;
        worker.abort();
        assert!(worker.await.unwrap_err().is_cancelled());
        assert!(matches!(
            waiting.await,
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        ));
        assert_eq!(queue.waiting_slots.available_permits(), 1);
    }
}
