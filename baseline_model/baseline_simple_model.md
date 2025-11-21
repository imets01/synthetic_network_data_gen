About the baseline model:

We though that we need to generate data the simplest way possible first, so at least we have one solution, if we don't get anywhere with the 2 step data generation.

The goal is to create a high-level feature set that is simple enough for a CTGAN to learn effectively, but descriptive enough to act as a powerful "blueprint" for a rule-based script to generate the packets. So in this approach the low level features will be generated thorugh a deterministic script.

- The GAN's Job (Stage 1): Generate the story of a connection. "This was a 30-second connection that migrated after 10 seconds, used 5 streams, and was closed by the client." It learns the realistic combinations of these story points.
- The Script's Job (Stages 2 & 3 combined): Act as a protocol-aware "director" that takes the GAN's story and generates the exact sequence of packets (the "movie") that tells that story, ensuring every packet follows the rules of QUIC.

Based on research we did, we shouldn't include features for the GAN which are too detailed, e.g. ACK count, because that depends on how many packets there are sent in the flow.

Features for the simple baseline model:

 - handshake_duration_msec
 - connection_duration_msec: Total duration of the flow in seconds.
 - total_client_app_bytes: Total application bytes sent by the client. Dictates how many STREAM frames to create.
 - total_server_app_bytes: Total application bytes sent by the server. Dictates how many STREAM frames to create.
 - client_bidi_streams_count
 - client_uni_streams_count
 - server_uni_streams_count
 - retry_occurred
 - migration_type
 - time_to_migration_msec
 - connection_close_type
 - avg_request_size
 - avg_response_size
 - app_data_bytes_before_migration
 - server_issued_cid_count
 - migration_validation_duration_msec: Round-Trip Time (RTT) of the new path from the client's perspective.

 We don't include features like 'packet_count', as that is mostly the result of all the other features. So what we are doing instead is that we create statistical profiles for the different packets, e.g. The packets that contain ACK only, have a size of x bytes on average. We use these statistics to generate a realistic low level capture.
