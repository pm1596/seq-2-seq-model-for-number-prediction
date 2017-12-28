
# coding: utf-8

# In[59]:


from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import helpers
tf.reset_default_graph()
sess=tf.InteractiveSession()
tf.__version__


# In[46]:


PAD=0
EOS=1

vocab_size=10
input_embedding_size=20

encoder_hidden_units=20
decoder_hidden_units=encoder_hidden_units*2


# In[47]:


encoder_inputs=tf.placeholder(shape=(None,None),dtype=tf.int32,name='encode_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


# In[48]:


embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)


# In[49]:



from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple


# In[50]:


encoder_cell = LSTMCell(encoder_hidden_units)


# In[51]:


((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float64, time_major=True)
    )


# In[60]:


encoder_fw_outputs
encoder_bw_outputs
encoder_bw_final_state


# In[61]:


encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
    
encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
    
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)


# In[62]:


decoder_cell = LSTMCell(decoder_hidden_units)


# In[63]:


encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
decoder_lengths = encoder_inputs_length + 3


# In[64]:


W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)

b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)


# In[65]:


assert EOS==1 and PAD==0

eos_time_slice=tf.ones([batch_size],dtype=tf.int32,name='EOS')
pad_time_slice=tf.zeros([batch_size],dtype=tf.int32,name='PAD')

eos_step_embedded=tf.nn.embedding_lookup(embeddings,eos_time_slice)
pad_step_embedded=tf.nn.embedding_lookup(embeddings,pad_time_slice)


# In[66]:


def loop_fun_initial():
    initial_elements_finished=(0>=decoder_lengths)
    
    initial_input=eos_step_embedded
    
    initial_cell_state=None
    
    initial_loop_state=None
    
    return(initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_loop_state)


# In[67]:


def loop_fun_transition(time,previous_output,previous_state,previous_loop_state):
    def generate_next_input():
        output_logits=tf.add(tf.matmul(previous_output,W),b)
        
        prediction=tf.argmax(output_logits,axis=1)
        
        next_input=tf.nn.embedding_lookup(embeddings,prediction)
        return next_input
        
    elements_finished=(time>=decoder_lengths)
    
    finished=tf.reduce_all(elements_finished)
    input=tf.cond(finished,lambda:pad_steps_embedded,generate_next_input)
    state=previous_state
    output=previous_output
    loop_state=None
    
    return(elements_finished,
            input,
            state,
            output,
            loop_space)


# In[39]:


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fun_initial()
    else:
        return loop_fun_transition(time, previous_output, previous_state, previous_loop_state)
        
decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()


# In[68]:


decoder_outputs


# In[69]:


decoder_max_steps,decoder_batch_size,decoder_dim=tf.unstack(tf.shape(decoder_outputs))

decoder_outputs_flat=tf.reshape(decoder_outputs,(-1,decoder_dim))
decoder_logits_flat=tf.add(tf.matmul(decoder_outputs_flat,W),b)
decoder_logits=tf.reshape(decoder_logits_flat,(decoder_max_steps,decoder_batch_size,vocab_size))


# In[70]:


decoder_prediction = tf.argmax(decoder_logits, 2)


# In[71]:


stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits
)

loss=tf.reduce_mean(stepwise_cross_entropy)

train_op=tf.train.AdamOptimizer().minimize(loss)


# In[72]:


sess.run(tf.global_variables_initializer())


# In[73]:


batch_size=50

batches=helpers.random_sequences(length_from=3, length_to=8,
                                    vocab_lower=2,vocab_upper=10,
                                    batch_size=batch_size)
print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)


# In[85]:


def next_feed():
    batch=next(batches)
    encoder_inputs_,encoder_input_lengths_=helpers.batch(batch)
    decoder_targets_=helpers.batch(
            [(sequence)+[EOS]+[PAD]*2 for sequence in batch]
    )
    return{
         encoder_inputs: encoder_inputs_,
         encoder_inputs_length: encoder_inputs_lengths_,
         decoder_targets: decoder_targets_
    }


# In[86]:


loss_track=[]


# In[87]:


max_batches=10
batches_in_epoch=2

try:
    for batch in range(max_batches):
        fd = next_feed()
        l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('trainig interrupts')
    


# In[76]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))





