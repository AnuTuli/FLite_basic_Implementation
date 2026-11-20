import os
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization

aes_key = os.urandom(16)

with open("peer_public.pem", "rb") as f:
    recipient_public_key = serialization.load_pem_public_key(f.read())

encrypted_aes_key = recipient_public_key.encrypt(
    aes_key,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)

with open("encrypted_aes_key.bin", "wb") as f:
    f.write(encrypted_aes_key)

print("AES-128 key encrypted successfully.")
print("Encrypted key size:", len(encrypted_aes_key), "bytes")
