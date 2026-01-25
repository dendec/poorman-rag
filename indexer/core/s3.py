import os
import boto3
import logging
from typing import Optional

logger = logging.getLogger("indexer.s3")

class S3Uploader:
    def __init__(
        self, 
        bucket: str, 
        endpoint: Optional[str] = None, 
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ):
        self.bucket = bucket
        
        # Initialize boto3 client with optional custom endpoint and keys
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        logger.info(f"S3 Client initialized (Bucket: {bucket}, Endpoint: {endpoint or 'AWS Native'})")

    def upload_file(self, local_path: str, s3_path: str):
        """Uploads a local file to the specified S3 path."""
        if not os.path.exists(local_path):
            logger.error(f"Local file not found: {local_path}")
            return False
            
        try:
            logger.info(f"Uploading {local_path} -> s3://{self.bucket}/{s3_path}")
            self.client.upload_file(local_path, self.bucket, s3_path)
            logger.info("✅ Upload successful")
            return True
        except Exception as e:
            logger.error(f"💥 Upload failed: {e}", exc_info=True)
            return False

def get_uploader_from_config(config) -> Optional[S3Uploader]:
    """Helper to create uploader if S3 bucket is specified in config."""
    if not hasattr(config, "s3_bucket") or not config.s3_bucket:
        return None
        
    return S3Uploader(
        bucket=config.s3_bucket,
        endpoint=config.s3_endpoint,
        region=config.s3_region,
        access_key=config.s3_access_key,
        secret_key=config.s3_secret_key
    )