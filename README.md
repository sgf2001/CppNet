CppNet

CppNet is a purpose-built tool designed for the identification of cell-penetrating peptides (CPPs) that extract features directly from sequences. CppNet's protein pre-training model and molecular fingerprint based on Transformer architecture extract the global and local features of the sequence to predict Cpp, respectively.

Pretrtain code

For the ESM-2 pre-trained model, the "esm2_t33_650M_UR50D" parameter was used to pre-train the peptide sequence, please visit: https://github.com/facebookresearch/esm for details

Run code

To train CppNet, you need to run the pretrain.py to get the pre-trained model embedded, and then run the main.py to train the model
