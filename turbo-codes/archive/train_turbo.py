import tensorflow as tf
print(tf.executing_eagerly())
from tensorflow import keras
import argparse


from src.models import nonsyscnn4_1000_22221_5m_shared, syscnn4_1000_22221_5m_shared, syscnn3_1000_33331_9m_shared, syscnn1_1000_2m_shared, syscnn0_2m_shared, syscnn1_1000_4m_shared, syscnn1_1000_4m_separate, syscnn2_1000_33_5m_shared, syscnn2_708_33_5m_separate, syscnn3_1000_333_7m_shared, syscnn3_1000_551_9m_shared, syscnn2_1000_3311_5m_shared
from src.data import data_generator
from src.metrics import bit_error_rate, block_error_rate

# tf.debugging.experimental.enable_dump_debug_info(
#     "./tmp/tfdbg2_logdir",
#     tensor_debug_mode="FULL_HEALTH",
#     circular_buffer_size=-1)

batch_size=500
block_len=100
steps_per_epoch=100
epochs=1000
snr = 0.0
num_decode_iter=6

tensorboard_callback = keras.callbacks.TensorBoard(
    './logs',
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq="epoch",
    profile_batch=2,
    embeddings_freq=0,
    embeddings_metadata=None
)
lr_decreaser_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=10, verbose=2,
    mode='min', min_delta=0.0001, cooldown=20, min_lr=0
)

# opt = keras.optimizers.Adam(learning_rate=0.0001)
model = nonsyscnn4_1000_22221_5m_shared(snr=snr, block_len=block_len, num_decode_iter=num_decode_iter, batch_size=batch_size)
opt = keras.optimizers.SGD(learning_rate=0.001)
bce = keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=bce, metrics=[bit_error_rate, block_error_rate], run_eagerly=False)

model.summary()

model.fit(
    x=data_generator(batch_size, block_len, model),
    epochs=epochs,
    verbose=1,
    callbacks=[tensorboard_callback, lr_decreaser_callback],
    steps_per_epoch=steps_per_epoch,
)
# Can't save for now because we don't have configs for layers
# model.save(f'./models/syscnn1_1000_2m_shared_snr{int(snr)}_decode{num_decode_iter}.h5')