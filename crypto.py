
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes


def generate_aes128_key():

    return AESGCM.generate_key(bit_length=128)


def encrypt_local_message(message, aes_key):

    nonce = os.urandom(12)

    aes = AESGCM(aes_key)

    ciphertext = aes.encrypt(
        nonce,
        message.encode("utf-8"),
        None
    )

    return nonce, ciphertext


def decrypt_local_message(nonce, ciphertext, aes_key):

    aes = AESGCM(aes_key)

    plaintext = aes.decrypt(
        nonce,
        ciphertext,
        None
    )

    return plaintext.decode("utf-8")


def generate_rsa_key_pair():

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )

    public_key = private_key.public_key()

    return private_key, public_key


def wrap_aes_key(aes_key, public_key):

    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(
                algorithm=hashes.SHA256()
            ),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return encrypted_key


def unwrap_aes_key(encrypted_key, private_key):

    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(
                algorithm=hashes.SHA256()
            ),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return aes_key


if __name__ == "__main__":

    message = "Local AES-128-GCM demonstration."

    aes_key = generate_aes128_key()

    nonce, ciphertext = encrypt_local_message(
        message, aes_key
    )

    recovered_message = decrypt_local_message(
        nonce, ciphertext, aes_key
    )

    print("Original:", message)
    print("Recovered:", recovered_message)

    private_key, public_key = generate_rsa_key_pair()

    wrapped_key = wrap_aes_key(
        aes_key,
        public_key
    )

    recovered_key = unwrap_aes_key(
        wrapped_key,
        private_key
    )

    print("AES key successfully recovered:",
          aes_key == recovered_key)
