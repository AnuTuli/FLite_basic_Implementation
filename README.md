Five edge nodes are connected in peer-to-peer form.
Each edge node encrypts AES-128-GCM key using RSA public key of the receiver and exchanges with neighbours.
The receiver after receiving the encrypted key decrypts it using own private key. 
Each edge node trains its model using transformer, encrypts model weights and exchanges with neighbours.
Each edge node decrypts the received model weights and aggregates them along with its own model weight.
The process continues for a number of rounds until each edge node has a model with minimal loss.
