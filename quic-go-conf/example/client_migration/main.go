package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"flag"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/quic-go/quic-go"
	"github.com/quic-go/quic-go/http3"
	"github.com/quic-go/quic-go/http3/qlog"
	"github.com/quic-go/quic-go/internal/testdata"
)

func main() {
	// --- FLAGS -------------------------------------------------------------
	quiet := flag.Bool("q", false, "don't print the data")
	keyLogFile := flag.String("keylog", "", "key log file")
	insecure := flag.Bool("insecure", false, "skip certificate verification")

	sourceIP := flag.String("source-ip", "127.0.0.2", "source IP before migration")
	newIP := flag.String("new-ip", "127.0.0.3", "destination IP after migration")
	performMigration := flag.Bool("perform-migration", false, "perform IP migration")

	flag.Parse()
	urls := flag.Args()
	if len(urls) == 0 {
		log.Fatal("no URL specified")
	}

	// --- KEYLOG ------------------------------------------------------------
	var keyLog io.Writer
	if len(*keyLogFile) > 0 {
		// Append mode so multiple runs can share the same file
		f, err := os.OpenFile(*keyLogFile, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0600)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		keyLog = f
	}

	// --- TLS ROOT CAs ------------------------------------------------------
	pool, err := x509.SystemCertPool()
	if err != nil {
		log.Fatal(err)
	}
	testdata.AddRootCA(pool)

	tlsConf := &tls.Config{
		RootCAs:            pool,
		InsecureSkipVerify: *insecure,
		KeyLogWriter:       keyLog,
		// ServerName must match the certificate
		ServerName: "localhost",
	}

	quicConf := &quic.Config{
		Tracer: qlog.DefaultConnectionTracer,
	}

	// --- HTTP/3 TRANSPORT + CUSTOM DIAL (LOW-LEVEL MIGRATION) -------------
	roundTripper := &http3.Transport{
		TLSClientConfig: tlsConf,
		QUICConfig:      quicConf,
	}

	roundTripper.Dial = func(
		ctx context.Context,
		addr string,
		tlsConf *tls.Config,
		quicConf *quic.Config,
	) (*quic.Conn, error) {
		raddr, err := net.ResolveUDPAddr("udp", addr)
		if err != nil {
			return nil, err
		}

		// 1) Setup Path 1
		laddr1 := &net.UDPAddr{IP: net.ParseIP(*sourceIP), Port: 0}
		udpConn1, err := net.ListenUDP("udp", laddr1)
		if err != nil {
			return nil, err
		}
		tr1 := &quic.Transport{Conn: udpConn1}

		// Dial returns as soon as the connection is established (before handshake completes fully)
		conn, err := tr1.Dial(ctx, raddr, tlsConf, quicConf)
		if err != nil {
			return nil, err
		}

		// --- BLOCKING CHANGE STARTS HERE ---

		// 2) Wait for Handshake to complete BEFORE doing anything else
		// We need the handshake done so we have keys to migrate.
		select {
		case <-conn.HandshakeComplete():
			log.Printf("Handshake complete on %s", laddr1)
		case <-ctx.Done():
			return nil, ctx.Err()
		}

		// 3) Perform Migration (Synchronously)
		// We make the HTTP client wait.
		if *performMigration {
			log.Printf("Starting pre-request migration...")

			// Setup Path 2
			laddr2 := &net.UDPAddr{IP: net.ParseIP(*newIP), Port: 0}
			udpConn2, err := net.ListenUDP("udp", laddr2)
			if err != nil {
				return nil, err
			}
			tr2 := &quic.Transport{Conn: udpConn2}

			log.Printf("Adding path...")
			path, err := conn.AddPath(tr2)
			if err != nil {
				return nil, err
			}

			// Probe is REQUIRED for quic-go to validate the path
			log.Printf("Probing path...")
			ctxProbe, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := path.Probe(ctxProbe); err != nil {
				log.Printf("Probe failed: %v", err)
				return nil, err
			}

			log.Printf("Switching path...")
			if err := path.Switch(); err != nil {
				return nil, err
			}
			log.Printf("Switch successful! Now on %s", laddr2)
		}

		// 4) Return the connection ONLY after migration is finished.
		// The HTTP Client will now send the GET request over the NEW IP.
		return conn, nil
	}

	defer roundTripper.Close()
	hclient := &http.Client{Transport: roundTripper}

	var wg sync.WaitGroup
	wg.Add(len(urls))
	for _, addr := range urls {
		log.Printf("GET %s", addr)
		go func(addr string) {
			defer wg.Done()

			resp, err := hclient.Get(addr)
			if err != nil {
				log.Fatal(err)
			}
			defer resp.Body.Close()

			log.Printf("Got response for %s: %#v", addr, resp)
			body := &bytes.Buffer{}
			_, err = io.Copy(body, resp.Body)
			if err != nil {
				log.Fatal(err)
			}

			if *quiet {
				log.Printf("Response Body: %d bytes", body.Len())
			} else {
				log.Printf("Response Body (%d bytes):\n%s", body.Len(), body.Bytes())
			}

		}(addr)
	}
	wg.Wait()
}
