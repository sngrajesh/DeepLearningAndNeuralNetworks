# TensorFlow Reference

## Core Libraries and Modules

### tf.keras
* Models
  * `tf.keras.Model`
  * `tf.keras.Sequential`
  * `tf.keras.model.save(filepath, overwrite=True, save_format=None)`
  * `tf.keras.models.load_model(filepath)`

* Layers
  * Dense
    * `tf.keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform')`
  * Convolutional
    * `tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid')`
    * `tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid')`
    * `tf.keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid')`
  * Recurrent
    * `tf.keras.layers.LSTM(units, activation='tanh', recurrent_activation='sigmoid')`
    * `tf.keras.layers.GRU(units, activation='tanh', recurrent_activation='sigmoid')`
    * `tf.keras.layers.SimpleRNN(units, activation='tanh')`
  * Pooling
    * `tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')`
    * `tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')`
    * `tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')`
  * Dropout
    * `tf.keras.layers.Dropout(rate, noise_shape=None, seed=None)`
  * Normalization
    * `tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)`
  * Reshaping
    * `tf.keras.layers.Flatten(data_format=None)`
    * `tf.keras.layers.Reshape(target_shape)`
    * `tf.keras.layers.Permute(dims)`

* Optimizers
  * `tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)`
  * `tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)`
  * `tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)`
  * `tf.keras.optimizers.Adagrad(learning_rate=0.01)`
  * `tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)`

* Loss Functions
  * `tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)`
  * `tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0)`
  * `tf.keras.losses.MeanSquaredError()`
  * `tf.keras.losses.MeanAbsoluteError()`
  * `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)`

* Metrics
  * `tf.keras.metrics.Accuracy()`
  * `tf.keras.metrics.Precision()`
  * `tf.keras.metrics.Recall()`
  * `tf.keras.metrics.AUC()`
  * `tf.keras.metrics.MeanSquaredError()`

* Callbacks
  * `tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False)`
  * `tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)`
  * `tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)`
  * `tf.keras.callbacks.TensorBoard(log_dir='./logs')`

### tf.data
* Dataset Creation
  * `tf.data.Dataset.from_tensor_slices(tensors)`
  * `tf.data.Dataset.from_generator(generator, output_types, output_shapes=None)`
  * `tf.data.Dataset.list_files(file_pattern, shuffle=None, seed=None)`

* Dataset Operations
  * `dataset.shuffle(buffer_size, seed=None, reshuffle_each_iteration=None)`
  * `dataset.batch(batch_size, drop_remainder=False)`
  * `dataset.repeat(count=None)`
  * `dataset.map(map_func, num_parallel_calls=None)`
  * `dataset.prefetch(buffer_size)`
  * `dataset.filter(predicate)`

### tf.image
* Image Operations
  * `tf.image.resize(images, size, method=ResizeMethod.BILINEAR)`
  * `tf.image.flip_left_right(image)`
  * `tf.image.random_brightness(image, max_delta, seed=None)`
  * `tf.image.random_contrast(image, lower, upper, seed=None)`
  * `tf.image.random_crop(value, size, seed=None)`

### tf.math
* Basic Operations
  * `tf.add(x, y)`
  * `tf.subtract(x, y)`
  * `tf.multiply(x, y)`
  * `tf.divide(x, y)`
  * `tf.pow(x, y)`
  * `tf.sqrt(x)`
  * `tf.matmul(a, b)`

* Advanced Math
  * `tf.reduce_mean(input_tensor, axis=None)`
  * `tf.reduce_sum(input_tensor, axis=None)`
  * `tf.reduce_max(input_tensor, axis=None)`
  * `tf.argmax(input, axis=None)`
  * `tf.nn.softmax(logits, axis=None)`

### tf.random
* Random Number Generation
  * `tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)`
  * `tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.dtypes.float32)`
  * `tf.random.shuffle(value, seed=None)`

### tf.saved_model
* Model Saving/Loading
  * `tf.saved_model.save(obj, export_dir, signatures=None)`
  * `tf.saved_model.load(export_dir)`

## Preprocessing Tools

### tf.keras.preprocessing
* Image
  * `tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=0,
      width_shift_range=0.0,
      height_shift_range=0.0,
      brightness_range=None,
      zoom_range=0.0,
      horizontal_flip=False,
      vertical_flip=False,
      rescale=None
    )`
* Text
  * `tf.keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)`
  * `tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, padding='pre')`

## Feature Columns (for Structured Data)

### tf.feature_column
* Numeric Columns
  * `tf.feature_column.numeric_column(key, shape=(1,), default_value=None)`
* Categorical Columns
  * `tf.feature_column.categorical_column_with_vocabulary_list(key, vocabulary_list)`
  * `tf.feature_column.categorical_column_with_hash_bucket(key, hash_bucket_size)`
* Embedding Columns
  * `tf.feature_column.embedding_column(categorical_column, dimension)`
* Crossed Columns
  * `tf.feature_column.crossed_column(keys, hash_bucket_size)`

## Transfer Learning

### tf.keras.applications
* Pre-trained Models
  * `tf.keras.applications.ResNet50(weights='imagenet', include_top=True)`
  * `tf.keras.applications.VGG16(weights='imagenet', include_top=True)`
  * `tf.keras.applications.VGG19(weights='imagenet', include_top=True)`
  * `tf.keras.applications.InceptionV3(weights='imagenet', include_top=True)`
  * `tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)`
  * `tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=True)`

## Training Functions and Methods

### Model Training
* `model.compile(optimizer, loss, metrics=None)`
* `model.fit(x, y, batch_size=None, epochs=1, validation_data=None)`
* `model.evaluate(x, y, batch_size=None)`
* `model.predict(x, batch_size=None)`
* `model.summary()`

### Custom Training Loop
* `tf.GradientTape()`
* `tape.gradient(target, sources)`
* `optimizer.apply_gradients(grads_and_vars)`

## GPU Configuration

### tf.config
* `tf.config.list_physical_devices('GPU')`
* `tf.config.experimental.set_memory_growth(device, True)`
* `tf.config.set_visible_devices(devices, device_type)`

## Debugging and Monitoring

### tf.debugging
* `tf.debugging.assert_equal(x, y, message=None)`
* `tf.debugging.assert_greater(x, y, message=None)`
* `tf.debugging.assert_shapes(shapes)`

### tf.summary (for TensorBoard)
* `tf.summary.scalar(name, data, step=None)`
* `tf.summary.histogram(name, data, step=None)`
* `tf.summary.image(name, data, step=None)`