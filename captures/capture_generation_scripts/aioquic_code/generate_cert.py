import datetime
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def generate_self_signed_cert():
    """
    Generates a self-signed certificate and a private key.
    Saves them as 'ssl_cert.pem' and 'ssl_key.pem'.
    """
    # 1. Generate a private key
    private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    )
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"My Test Company"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
    ])

    # 3. Build the certificate
    cert_builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(
            # Certificate will be valid for 1 year
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        )
        .add_extension(
            # Basic constraints, marking it as a CA cert (common for self-signed)
            x509.BasicConstraints(ca=True, path_length=None), critical=True,
        )
    )

    # 4. Sign the certificate with the private key
    certificate = cert_builder.sign(private_key, hashes.SHA256())

    # 5. Save the private key to a file
    with open("ssl_key.pem", "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    print("Successfully generated 'ssl_key.pem'")

    # 6. Save the certificate to a file
    with open("ssl_cert.pem", "wb") as f:
        f.write(certificate.public_bytes(serialization.Encoding.PEM))
    print("Successfully generated 'ssl_cert.pem'")

if __name__ == "__main__":
    generate_self_signed_cert()
