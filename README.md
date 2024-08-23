CppNet

CppNet is a purpose-built tool designed for predictive cell-penetrating peptides (CPPs) that extract features directly from sequences. CppNet mainly extracts the global and local features of the sequence to predict Cpps through the pre-trained model based on the Transformer architecture and molecular fingerprint, respectively.

Pretrain code

For the ESM-2 pretrain model, the "esm2_t33_650M_UR50D" parameter was used to pre-train the peptide sequence, please visit: https://github.com/facebookresearch/esm for details.

Run code

To train CppNet, you need to run the pretrain.py to get the pre-trained model embedded, and then run the main.py to train the model.
