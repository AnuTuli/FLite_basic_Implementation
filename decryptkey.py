from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization

with open("peer_private.pem", "rb") as f:
    receiver_private_key = serialization.load_pem_private_key(
        f.read(),
        password=None
    )

with open("encrypted_aes_key.bin", "rb") as f:
    received_encrypted_key = f.read()

decrypted_aes_key = receiver_private_key.decrypt(
    received_encrypted_key,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)

print("AES key decrypted successfully.")
print("AES key length:", len(decrypted_aes_key), "bytes")
print("AES-128:", len(decrypted_aes_key) == 16)
