Five edge nodes are connected in peer-to-peer form.
Each edge node encrypts AES-128-GCM key using RSA public key of the receiver and exchanges with neighbours.
The receiver after receiving the encrypted key decrypts it using own private key. 
Each edge node trains its model using transformer, encrypts model weights and exchanges with neighbours.
Each edge node decrypts the received model weights and aggregates them along with its own model weight.
The process continues for a number of rounds until each edge node has a model with minimal loss.

The simulation codes for the P2P-based decentralized federated learning framework with Stratified distribution, Dirichlet distribution, and region-wise distribution (as crop type varies based on regions) of the considered dataset.
We have also provided the code of one peer considered in our experiment along with its configuration, local training, and aggregation, in the P2P-based decentralized federated learning framework. 
In both simulation and experiment, we have used AES-128-GCM for encrypting and decrypting model weights during transmission.
