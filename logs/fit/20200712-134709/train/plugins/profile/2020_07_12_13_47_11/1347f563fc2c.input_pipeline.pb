	�&�O�`@�&�O�`@!�&�O�`@	�f�?���?�f�?���?!�f�?���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�&�O�`@3�xy�?A#����_@Y���7��?*	��S�+k@2F
Iterator::Model
-����?!"�R��J@)�׺���?1�F<�tF@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateB?S�[�?!m�c��:@)N��ĭ�?1� ��7@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�ߡ(�'�?!& �LKP0@)!ɬ��v�?1��h��-@:Preprocessing2S
Iterator::Model::ParallelMap�T���N�?!y!�AN@)�T���N�?1y!�AN@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip��h�x��?!�T�)�G@)pw�n��|?1��Ʌ]�	@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��Q,��z?!r#�"�@)��Q,��z?1r#�"�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensoriQ��k?!&�7�)P�?)iQ��k?1&�7�)P�?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapT�T�	g�?!�v�9{7<@) a��*f?1�.k-��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	3�xy�?3�xy�?!3�xy�?      ��!       "      ��!       *      ��!       2	#����_@#����_@!#����_@:      ��!       B      ��!       J	���7��?���7��?!���7��?R      ��!       Z	���7��?���7��?!���7��?JCPU_ONLY