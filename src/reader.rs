use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use tokio::sync::Mutex;

use crate::error::{HDF5Error, Result};

/// Async interface for reading byte ranges from an HDF5 file.
///
/// Modeled after async-tiff's `AsyncFileReader` trait. Implementations exist
/// for `object_store::ObjectStore`, `reqwest`, and `tokio::fs::File`.
#[async_trait]
pub trait AsyncFileReader: Debug + Send + Sync + 'static {
    /// Fetch the bytes in the given range.
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes>;

    /// Fetch multiple byte ranges. The default implementation calls `get_bytes`
    /// sequentially; `ObjectReader` overrides this with `get_ranges()`.
    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        let mut result = Vec::with_capacity(ranges.len());
        for range in ranges {
            let data = self.get_bytes(range).await?;
            result.push(data);
        }
        Ok(result)
    }
}

#[async_trait]
impl AsyncFileReader for Box<dyn AsyncFileReader + '_> {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        self.as_ref().get_bytes(range).await
    }

    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        self.as_ref().get_byte_ranges(ranges).await
    }
}

#[async_trait]
impl AsyncFileReader for Arc<dyn AsyncFileReader + '_> {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        self.as_ref().get_bytes(range).await
    }

    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        self.as_ref().get_byte_ranges(ranges).await
    }
}

// ── ObjectReader ────────────────────────────────────────────────────────────

/// An AsyncFileReader that reads from an [`ObjectStore`](object_store::ObjectStore).
#[cfg(feature = "object_store")]
#[derive(Clone, Debug)]
pub struct ObjectReader {
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
}

#[cfg(feature = "object_store")]
impl ObjectReader {
    /// Create a new ObjectReader.
    pub fn new(store: Arc<dyn object_store::ObjectStore>, path: object_store::path::Path) -> Self {
        Self { store, path }
    }
}

#[cfg(feature = "object_store")]
#[async_trait]
impl AsyncFileReader for ObjectReader {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        use object_store::ObjectStoreExt;

        self.store
            .get_range(&self.path, range)
            .await
            .map_err(HDF5Error::from)
    }

    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        self.store
            .get_ranges(&self.path, &ranges)
            .await
            .map_err(HDF5Error::from)
    }
}

// ── TokioReader ─────────────────────────────────────────────────────────────

/// An AsyncFileReader that wraps a `tokio::fs::File` or similar async reader.
#[cfg(feature = "tokio")]
#[derive(Debug)]
pub struct TokioReader<T: tokio::io::AsyncRead + tokio::io::AsyncSeek + Unpin + Send + Debug>(
    tokio::sync::Mutex<T>,
);

#[cfg(feature = "tokio")]
impl<T: tokio::io::AsyncRead + tokio::io::AsyncSeek + Unpin + Send + Debug> TokioReader<T> {
    /// Create a new TokioReader.
    pub fn new(inner: T) -> Self {
        Self(tokio::sync::Mutex::new(inner))
    }
}

#[cfg(feature = "tokio")]
#[async_trait]
impl<T: tokio::io::AsyncRead + tokio::io::AsyncSeek + Unpin + Send + Debug + 'static>
    AsyncFileReader for TokioReader<T>
{
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        use std::io::SeekFrom;
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        let mut file = self.0.lock().await;
        file.seek(SeekFrom::Start(range.start)).await?;

        let to_read = (range.end - range.start) as usize;
        let mut buffer = vec![0u8; to_read];

        // Use read_buf loop instead of read_exact to handle EOF gracefully.
        // The BlockCache may request ranges past the end of the file.
        let mut total_read = 0;
        while total_read < to_read {
            let n = file.read(&mut buffer[total_read..]).await?;
            if n == 0 {
                break; // EOF
            }
            total_read += n;
        }
        buffer.truncate(total_read);
        Ok(buffer.into())
    }
}

// ── ReqwestReader ───────────────────────────────────────────────────────────

/// An AsyncFileReader that reads from a URL using reqwest HTTP range requests.
#[cfg(feature = "reqwest")]
#[derive(Debug, Clone)]
pub struct ReqwestReader {
    client: reqwest::Client,
    url: reqwest::Url,
}

#[cfg(feature = "reqwest")]
impl ReqwestReader {
    /// Create a new ReqwestReader.
    pub fn new(client: reqwest::Client, url: reqwest::Url) -> Self {
        Self { client, url }
    }
}

#[cfg(feature = "reqwest")]
#[async_trait]
impl AsyncFileReader for ReqwestReader {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        let range_header = format!("bytes={}-{}", range.start, range.end - 1);
        let response = self
            .client
            .get(self.url.clone())
            .header("Range", range_header)
            .send()
            .await?
            .error_for_status()?;
        let bytes = response.bytes().await?;
        Ok(bytes)
    }
}

// ── BlockCache ──────────────────────────────────────────────────────────────

/// A caching wrapper that fetches fixed-size aligned blocks around each
/// accessed offset.
///
/// Unlike a sequential readahead cache, a block cache handles the scattered
/// access patterns of HDF5 metadata efficiently: the superblock is at offset 0,
/// object headers and B-tree nodes are scattered across the file, and a
/// sequential cache would waste bandwidth fetching unneeded array data between
/// metadata structures.
///
/// When a byte range is requested, the cache fetches any aligned blocks that
/// overlap the range and caches them for future reads.  Nearby metadata reads
/// (e.g., an object header and its continuation chunk, or adjacent B-tree
/// nodes) naturally share blocks.
///
/// Default block size is 8 MiB, which typically resolves all metadata for a
/// GOES-16 MCMIPF file (~164 datasets) in about 10 requests.
#[derive(Debug)]
pub struct BlockCache<F: AsyncFileReader> {
    inner: F,
    blocks: Arc<Mutex<HashMap<u64, Bytes>>>,
    block_size: u64,
}

impl<F: AsyncFileReader> BlockCache<F> {
    /// Create a new BlockCache wrapping the given reader.
    /// Default block size: 8 MiB.
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            blocks: Arc::new(Mutex::new(HashMap::new())),
            block_size: 8 * 1024 * 1024,
        }
    }

    /// Access the inner reader.
    pub fn inner(&self) -> &F {
        &self.inner
    }

    /// Set the block size in bytes.
    pub fn with_block_size(mut self, block_size: u64) -> Self {
        self.block_size = block_size;
        self
    }

    /// Aligned block start for a given offset.
    fn block_start(&self, offset: u64) -> u64 {
        offset / self.block_size * self.block_size
    }
}

#[async_trait]
impl<F: AsyncFileReader + Send + Sync> AsyncFileReader for BlockCache<F> {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        let len = (range.end - range.start) as usize;
        if len == 0 {
            return Ok(Bytes::new());
        }

        // Determine which blocks we need.
        let first_block = self.block_start(range.start);
        let last_block = self.block_start(range.end.saturating_sub(1));

        // Fast path: single block (most common case for metadata reads).
        if first_block == last_block {
            let block = self.ensure_block(first_block).await?;
            let local_start = (range.start - first_block) as usize;
            let local_end = local_start + len;
            // Handle reads near EOF where block may be shorter.
            let actual_end = local_end.min(block.len());
            if local_start >= block.len() {
                return Ok(Bytes::new());
            }
            return Ok(block.slice(local_start..actual_end));
        }

        // Multi-block: assemble from consecutive blocks.
        let mut out = BytesMut::with_capacity(len);
        let mut offset = range.start;
        let mut block_offset = first_block;

        while offset < range.end {
            let block = self.ensure_block(block_offset).await?;
            let local_start = (offset - block_offset) as usize;
            let bytes_from_block = ((block_offset + self.block_size) - offset)
                .min(range.end - offset) as usize;
            let actual_end = (local_start + bytes_from_block).min(block.len());
            if local_start >= block.len() {
                break; // EOF
            }
            out.extend_from_slice(&block[local_start..actual_end]);
            if actual_end < local_start + bytes_from_block {
                break; // EOF mid-block
            }
            offset += bytes_from_block as u64;
            block_offset += self.block_size;
        }

        Ok(out.into())
    }

    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        let mut result = Vec::with_capacity(ranges.len());
        for range in ranges {
            result.push(self.get_bytes(range).await?);
        }
        Ok(result)
    }
}

impl<F: AsyncFileReader> BlockCache<F> {
    /// Ensure a block is in the cache, fetching it if not.
    async fn ensure_block(&self, block_start: u64) -> Result<Bytes> {
        {
            let cache = self.blocks.lock().await;
            if let Some(block) = cache.get(&block_start) {
                return Ok(block.clone());
            }
        }

        // Fetch the block.  The block may be shorter than block_size at EOF.
        let fetch_range = block_start..block_start + self.block_size;
        let data = self.inner.get_bytes(fetch_range).await?;

        let mut cache = self.blocks.lock().await;
        cache.insert(block_start, data.clone());
        Ok(data)
    }
}
