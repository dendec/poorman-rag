package utils

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/klauspost/compress/zstd"
)

// S3Options holds configuration for S3 client
type S3Options struct {
	Bucket    string
	Endpoint  string
	Region    string
	AccessKey string
	SecretKey string
}

// EnsureDownload downloads all specified keys in parallel to /tmp.
func EnsureDownload(opts S3Options, keys ...string) ([]string, error) {
	if len(keys) == 0 {
		return nil, nil
	}

	ctx := context.TODO()
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("aws config: %w", err)
	}

	// Modern way to set custom endpoint in AWS SDK v2
	s3Opts := []func(*s3.Options){}
	endpoint := os.Getenv("RAG_S3_ENDPOINT")
	if endpoint != "" {
		s3Opts = append(s3Opts, func(o *s3.Options) {
			o.BaseEndpoint = aws.String(endpoint)
			// For S3-compatible providers (R2, Minio), usually need this
			o.UsePathStyle = true
		})
	}

	region := os.Getenv("RAG_S3_REGION")
	if region != "" {
		cfg.Region = region
	}

	client := s3.NewFromConfig(cfg, s3Opts...)

	localPaths := make([]string, len(keys))
	var wg sync.WaitGroup
	errCh := make(chan error, len(keys))

	start := time.Now()

	for i, key := range keys {
		// If it's .zst, the local file will be without this extension
		fileName := filepath.Base(key)
		isZstd := filepath.Ext(fileName) == ".zst"
		if isZstd {
			fileName = fileName[:len(fileName)-len(".zst")]
		}

		localPath := filepath.Join("/tmp", fileName)
		localPaths[i] = localPath

		if fileExists(localPath) {
			slog.Debug("file already exists, skipping", "path", localPath)
			continue
		}

		wg.Add(1)
		go func(k, dst string, z bool) {
			defer wg.Done()
			if err := downloadOne(client, opts.Bucket, k, dst, z); err != nil {
				errCh <- fmt.Errorf("download %s: %w", k, err)
			}
		}(key, localPath, isZstd)
	}

	wg.Wait()
	close(errCh)

	for e := range errCh {
		if e != nil {
			return nil, e
		}
	}

	slog.Info("download finished", "duration", time.Since(start), "count", len(keys))
	return localPaths, nil
}

func downloadOne(svc *s3.Client, bucket, key, dst string, isZstd bool) error {
	slog.Info("downloading file", "key", key, "dest", dst, "is_zstd", isZstd)

	tmpDst := dst + ".tmp"
	resp, err := svc.GetObject(context.TODO(), &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	f, err := os.Create(tmpDst)
	if err != nil {
		return fmt.Errorf("create tmp: %w", err)
	}

	if isZstd {
		zrd, err := zstd.NewReader(resp.Body)
		if err != nil {
			f.Close()
			return fmt.Errorf("zstd reader: %w", err)
		}
		_, err = io.Copy(f, zrd)
		zrd.Close()
	} else {
		_, err = io.Copy(f, resp.Body)
	}

	f.Close()
	if err != nil {
		return fmt.Errorf("write error: %w", err)
	}

	// Atomically rename to the target file
	if err := os.Rename(tmpDst, dst); err != nil {
		return fmt.Errorf("rename error: %w", err)
	}

	return nil
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir() && info.Size() > 0
}
