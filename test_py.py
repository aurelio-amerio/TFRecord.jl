#%%
import tensorflow as tf
import numpy as np
#%%
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _serialize_example( map_array, dldS_array, params, Fiso, Agal):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'map': _float_feature(map_array),
        'dldS': _float_feature(dldS_array),
        'params': _float_feature(params),
        'Fiso': _float_feature([Fiso]),
        'Agal': _float_feature([Agal]),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
#%%
_serialize_example(np.zeros(10), np.zeros(2), np.zeros(2), 1.0, 1.0)
# %%

# %%
with open("test.bin", "rb") as f:
  data = f.read()

print(data)


# %%
raw_dataset = tf.data.TFRecordDataset("example.tfrecord")
raw_dataset
# %%
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    print(raw_record.numpy())
    example.ParseFromString(raw_record.numpy())
    print(example)
# %%

#%%
# raw=b'\nM\nK\n\x08feature1\x12\x05\x1a\x03\n\x01\x01\n\x08feature3\x12\t\n\x07\n\x05horse\n\x08feature2\x12\x05\x1a\x03\n\x01\x02\n\x08feature4\x12\x08\x12\x06\n\x04>\xb4\x87>'
raw=b'\nM\nK\n\x08feature1\x12\x05\x1a\x03\n\x01\x01\n\x08feature3\x12\t\n\x07\n\x05horse\n\x08feature2\x12\x05\x1a\x03\n\x01\x02\n\x08feature4\x12\x08\x12\x06\n\x04>\xb4\x87>'

# %%
example = tf.train.Example()
# %%
example.ParseFromString(raw)
# %%
print(example)
# %%
tf.io.read_file("example.tfrecord")
# %%
def _serialize_example_test(feature1, feature2):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'feature1': _int64_feature([feature1]),
        'feature2': _int64_feature([feature2]),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
# %%
with open("testpy.bin", "wb") as f:
  f.write(_serialize_example_test(420, 314))

# %%
# %%
b"\n(\n\x12\n\x08"
# %%
with open("test.bin", "rb") as f:
  data = f.read()

example = tf.train.Example()
example.ParseFromString(data)
# %%
print(example)
# %%
# serex =  _serialize_example_test(420, 314)
serex =  data
features  = {
    "feature1": tf.io.FixedLenFeature([], dtype=tf.int64),
    "feature2": tf.io.FixedLenFeature([], dtype=tf.int64),
}
tf.io.parse_example(
    serex,
    features = features)
# %%
