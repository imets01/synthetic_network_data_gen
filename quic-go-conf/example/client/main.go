package main

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/quic-go/quic-go"
	"github.com/quic-go/quic-go/http3"
	"github.com/quic-go/quic-go/http3/qlog"
	"github.com/quic-go/quic-go/internal/testdata"
)

func main() {
	// --- CLI flags ----------------------------------------------------------
	quiet := flag.Bool("q", false, "don't print the data")
	// Path to the TLS key log file (NSS format, used by Wireshark to decrypt QUIC)
	keyLogFile := flag.String("keylog", "", "key log file")
	// For local testing: skip certificate verification (DON'T use in production).
	insecure := flag.Bool("insecure", false, "skip certificate verification")

	flag.Parse()
	urls := flag.Args()
	if len(urls) == 0 {
		log.Fatal("no URL specified")
	}

	// --- Open key log file (if requested) -----------------------------------
	var keyLog io.Writer
	if len(*keyLogFile) > 0 {
		f, err := os.Create(*keyLogFile)
		if err != nil {
			log.Fatal(err)
		}
		// The file will be closed when main() exits.
		defer f.Close()
		keyLog = f
	}

	// --- TLS root CAs -------------------------------------------------------
	pool, err := x509.SystemCertPool()
	if err != nil {
		log.Fatal(err)
	}
	// Add the quic-go test CA certificates (for local self-signed server)
	testdata.AddRootCA(pool)

	// --- HTTP/3 RoundTripper with QUIC + TLS config ------------------------
	roundTripper := &http3.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs:            pool,
			InsecureSkipVerify: *insecure,
			// This writes TLS secrets in NSS key log format so that Wireshark
			// can decrypt the QUIC / HTTP3 traffic.
			KeyLogWriter: keyLog,
		},
		QUICConfig: &quic.Config{
			// Enable qlog tracing for debugging (optional).
			Tracer: qlog.DefaultConnectionTracer,
		},
	}
	defer roundTripper.Close()

	hclient := &http.Client{
		Transport: roundTripper,
	}

	// --- Issue one HTTP/3 GET per URL --------------------------------------
	var wg sync.WaitGroup
	wg.Add(len(urls))
	for _, addr := range urls {
		log.Printf("GET %s", addr)
		go func(addr string) {
			defer wg.Done()

			rsp, err := hclient.Get(addr)
			if err != nil {
				log.Fatal(err)
			}
			log.Printf("Got response for %s: %#v", addr, rsp)

			body := &bytes.Buffer{}
			_, err = io.Copy(body, rsp.Body)
			if err != nil {
				log.Fatal(err)
			}
			_ = rsp.Body.Close()

			if *quiet {
				log.Printf("Response Body: %d bytes", body.Len())
			} else {
				log.Printf("Response Body (%d bytes):\n%s", body.Len(), body.Bytes())
			}
		}(addr)
	}
	wg.Wait()
}
