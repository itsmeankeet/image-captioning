import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers, Model, metrics


# For serialization compatibility (TF 2.18+Keras 3.8)
from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Fixed length allowed for any sequence
SEQ_LENGTH = 24

# Vocabulary size
VOCAB_SIZE = 13000

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Batch size
BATCH_SIZE = 128
@register_keras_serializable()
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate)
    
    def get_config(self):
        return {
            'post_warmup_learning_rate': self.post_warmup_learning_rate,
            'warmup_steps': self.warmup_steps
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
def get_cnn_model():
    base_model = EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False, # Removing the prediction layers
        weights="imagenet")
    # Freezing the model's weights
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

@register_keras_serializable()
class TransformerEncoderBlock(layers.Layer): 
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs): 
        super().__init__(**kwargs) 
        self.embed_dim = embed_dim 
        self.dense_dim = dense_dim 
        self.num_heads = num_heads 
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.0) 
        self.layernorm_1 = layers.LayerNormalization() 
        self.layernorm_2 = layers.LayerNormalization() 
        self.dense_1 = layers.Dense(embed_dim, activation="relu") 
 
    def call(self, inputs, training, mask=None): 
        inputs = self.layernorm_1(inputs) 
        inputs = self.dense_1(inputs) 
        attention_output_1 = self.attention_1(query=inputs, 
                                           value=inputs, 
                                           key=inputs, 
                                           attention_mask=None, 
                                           training=training) 
        out_1 = self.layernorm_2(inputs + attention_output_1) 
        return out_1

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config


@register_keras_serializable()
class PositionalEmbedding(layers.Layer): 
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs): 
        super().__init__(**kwargs) 
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim) 
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim) 
        self.sequence_length = sequence_length 
        self.vocab_size = vocab_size 
        self.embed_dim = embed_dim 
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32)) 
 
    def call(self, inputs): 
        length = tf.shape(inputs)[-1] 
        positions = tf.range(start=0, limit=length, delta=1) # Positional encoding 
        embedded_tokens = self.token_embeddings(inputs) # Input embedding 
        embedded_tokens = embedded_tokens * self.embed_scale 
        embedded_positions = self.position_embeddings(positions) 
        return embedded_tokens + embedded_positions # Positional embedding 
 
    def compute_mask(self, inputs, mask=None): 
        return tf.math.not_equal(inputs, 0)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config


@register_keras_serializable()
class TransformerDecoderBlock(layers.Layer): 
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs): 
        super().__init__(**kwargs) 
        self.embed_dim = embed_dim 
        self.ff_dim = ff_dim 
        self.num_heads = num_heads 
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1) 
        self.cross_attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1) 
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu") 
        self.ffn_layer_2 = layers.Dense(embed_dim) 
 
        self.layernorm_1 = layers.LayerNormalization() 
        self.layernorm_2 = layers.LayerNormalization() 
        self.layernorm_3 = layers.LayerNormalization() 
 
        self.embedding = PositionalEmbedding(embed_dim=EMBED_DIM, 
                                          sequence_length=SEQ_LENGTH, 
                                          vocab_size=VOCAB_SIZE) 
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax") 
 
        self.dropout_1 = layers.Dropout(0.3) 
        self.dropout_2 = layers.Dropout(0.5) 
        self.supports_masking = True 
 
    def call(self, inputs, encoder_outputs, training, mask=None): 
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
    
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else:
            # No padding, so "mask everything as allowed"
            padding_mask = tf.ones((batch_size, seq_len, 1), dtype=tf.int32)
            combined_mask = causal_mask  # only causal mask applies
    
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)
        
        cross_attention_output_2 = self.cross_attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training
        )
        out_2 = self.layernorm_2(out_1 + cross_attention_output_2)
    
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds 
     
    def get_causal_attention_mask(self, inputs): 
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1),tf.constant([1, 1], dtype=tf.int32)],axis=0)
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "num_heads": self.num_heads
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.utils.register_keras_serializable()
class ImageCaptioningModel(keras.Model):
    def __init__(self, cnn_model, encoder, decoder, num_captions_per_image=5, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # Print model shapes for debugging
        print()
        print(f'CNN input shape: {cnn_model.input_shape}')
        print(f'CNN output shape: {cnn_model.output_shape}', end='\n'*2)
        print(f'Encoder input ---> Dense layer shape: {cnn_model.output_shape} ---> (None, {cnn_model.output_shape[1]}, {EMBED_DIM})')
        print(f'Encoder output shape: (None, {cnn_model.output_shape[1]}, {EMBED_DIM})', end='\n'*2)
        print(f'Decoder input 1 (Caption) ---> Positional Embedding shape: (None, {SEQ_LENGTH-1}) ---> (None, {SEQ_LENGTH-1}, {EMBED_DIM})')
        print(f'Decoder input 2 (Embedded image features) shape: (None, {cnn_model.output_shape[1]}, {EMBED_DIM})')
        print(f'Decoder output (MH Cross-Attention) shape: (None, {SEQ_LENGTH-1}, {EMBED_DIM})')
        print(f'Decoder prediction (Dense layer) shape: (None, {SEQ_LENGTH-1}, {VOCAB_SIZE})')

    def get_config(self):
        config = super().get_config()
        config.update({
            'cnn_model': keras.utils.serialize_keras_object(self.cnn_model),
            'encoder': keras.utils.serialize_keras_object(self.encoder),
            'decoder': keras.utils.serialize_keras_object(self.decoder),
            'image_aug': keras.utils.serialize_keras_object(self.image_aug) if self.image_aug else None,
            'num_captions_per_image': self.num_captions_per_image
        })
        return config

    @classmethod
    def from_config(cls, config):
        cnn_model = keras.utils.deserialize_keras_object(config['cnn_model'])
        encoder = keras.utils.deserialize_keras_object(config['encoder'])
        decoder = keras.utils.deserialize_keras_object(config['decoder'])
        image_aug = keras.utils.deserialize_keras_object(config['image_aug']) if config['image_aug'] else None
        num_captions_per_image = config['num_captions_per_image']
        return cls(
            cnn_model=cnn_model,
            encoder=encoder,
            decoder=decoder,
            num_captions_per_image = num_captions_per_image,
            image_aug=image_aug
        )

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=training, mask=mask)
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc
    
    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        if self.image_aug:
            batch_img = self.image_aug(batch_img)
        img_embed = self.cnn_model(batch_img)
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq[:, i, :], training=True)
                batch_loss += loss
                batch_acc += acc
            train_vars = (self.encoder.trainable_variables + self.decoder.trainable_variables)
            grads = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(grads, train_vars))
        batch_acc /= float(self.num_captions_per_image)
        # Return values directly, no tracker
        return {"loss": batch_loss, "acc": batch_acc}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        img_embed = self.cnn_model(batch_img)
        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq[:, i, :], training=False)
            batch_loss += loss
            batch_acc += acc
        batch_acc /= float(self.num_captions_per_image)
        return {"loss": batch_loss, "acc": batch_acc}

    def call(self, inputs, training=False):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            images, captions = inputs
        else:
            images = inputs
            captions = None
        img_embed = self.cnn_model(images, training=training)
        encoder_out = self.encoder(img_embed, training=training)
        if captions is not None:
            decoder_out = self.decoder(captions, encoder_out, training=training)
            return decoder_out
        else:
            return encoder_out

# No @property metrics needed!



# custom_objects = {
#     "ImageCaptioningModel": ImageCaptioningModel,
#     "LRSchedule": LRSchedule,
#     "TransformerEncoderBlock": TransformerEncoderBlock,
#     "TransformerDecoderBlock": TransformerDecoderBlock,
#     "PositionalEmbedding": PositionalEmbedding,
# }
# loading_model = tf.keras.models.load_model('./utils/best_model.keras', custom_objects=custom_objects, compile=False)

# print(loading_model.summary())