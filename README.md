# seq-2-seq-model-for-number-prediction

Encoder is bidirectional
Decoder is implemented using tf.nn.raw_rnn
Previously generated tokens are fed during training as inputs
encoder_inputs and decoder_inputs are int32 tensors of shape [max_time, batch_size]
The RNNs use backpropagation through time hence time and batch dimensions are dynamic
