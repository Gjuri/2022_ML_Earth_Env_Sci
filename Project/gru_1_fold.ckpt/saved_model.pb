Ą
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68�
�
Output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*$
shared_nameOutput_layer/kernel
{
'Output_layer/kernel/Read/ReadVariableOpReadVariableOpOutput_layer/kernel*
_output_shapes

:d*
dtype0
z
Output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput_layer/bias
s
%Output_layer/bias/Read/ReadVariableOpReadVariableOpOutput_layer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
Hidden_layer/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameHidden_layer/gru_cell/kernel
�
0Hidden_layer/gru_cell/kernel/Read/ReadVariableOpReadVariableOpHidden_layer/gru_cell/kernel*
_output_shapes
:	�*
dtype0
�
&Hidden_layer/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*7
shared_name(&Hidden_layer/gru_cell/recurrent_kernel
�
:Hidden_layer/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&Hidden_layer/gru_cell/recurrent_kernel*
_output_shapes
:	d�*
dtype0
�
Hidden_layer/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameHidden_layer/gru_cell/bias
�
.Hidden_layer/gru_cell/bias/Read/ReadVariableOpReadVariableOpHidden_layer/gru_cell/bias*
_output_shapes
:	�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/Output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/Output_layer/kernel/m
�
.Adam/Output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output_layer/kernel/m*
_output_shapes

:d*
dtype0
�
Adam/Output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_layer/bias/m
�
,Adam/Output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output_layer/bias/m*
_output_shapes
:*
dtype0
�
#Adam/Hidden_layer/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/Hidden_layer/gru_cell/kernel/m
�
7Adam/Hidden_layer/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/Hidden_layer/gru_cell/kernel/m*
_output_shapes
:	�*
dtype0
�
-Adam/Hidden_layer/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*>
shared_name/-Adam/Hidden_layer/gru_cell/recurrent_kernel/m
�
AAdam/Hidden_layer/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/Hidden_layer/gru_cell/recurrent_kernel/m*
_output_shapes
:	d�*
dtype0
�
!Adam/Hidden_layer/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*2
shared_name#!Adam/Hidden_layer/gru_cell/bias/m
�
5Adam/Hidden_layer/gru_cell/bias/m/Read/ReadVariableOpReadVariableOp!Adam/Hidden_layer/gru_cell/bias/m*
_output_shapes
:	�*
dtype0
�
Adam/Output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/Output_layer/kernel/v
�
.Adam/Output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output_layer/kernel/v*
_output_shapes

:d*
dtype0
�
Adam/Output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_layer/bias/v
�
,Adam/Output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output_layer/bias/v*
_output_shapes
:*
dtype0
�
#Adam/Hidden_layer/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/Hidden_layer/gru_cell/kernel/v
�
7Adam/Hidden_layer/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/Hidden_layer/gru_cell/kernel/v*
_output_shapes
:	�*
dtype0
�
-Adam/Hidden_layer/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*>
shared_name/-Adam/Hidden_layer/gru_cell/recurrent_kernel/v
�
AAdam/Hidden_layer/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/Hidden_layer/gru_cell/recurrent_kernel/v*
_output_shapes
:	d�*
dtype0
�
!Adam/Hidden_layer/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*2
shared_name#!Adam/Hidden_layer/gru_cell/bias/v
�
5Adam/Hidden_layer/gru_cell/bias/v/Read/ReadVariableOpReadVariableOp!Adam/Hidden_layer/gru_cell/bias/v*
_output_shapes
:	�*
dtype0

NoOpNoOp
�*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�*
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�
iter

beta_1

beta_2
	 decay
!learning_ratemLmM"mN#mO$mPvQvR"vS#vT$vU*
'
"0
#1
$2
3
4*
'
"0
#1
$2
3
4*
* 
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

*serving_default* 
�

"kernel
#recurrent_kernel
$bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses*
* 

"0
#1
$2*

"0
#1
$2*
* 
�

2states
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
c]
VARIABLE_VALUEOutput_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEOutput_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEHidden_layer/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&Hidden_layer/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEHidden_layer/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

=0
>1*
* 
* 
* 

"0
#1
$2*

"0
#1
$2*
* 
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
8
	Dtotal
	Ecount
F	variables
G	keras_api*
8
	Htotal
	Icount
J	variables
K	keras_api*
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

F	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

J	variables*
��
VARIABLE_VALUEAdam/Output_layer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/Output_layer/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/Hidden_layer/gru_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/Hidden_layer/gru_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/Hidden_layer/gru_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/Output_layer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/Output_layer/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/Hidden_layer/gru_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/Hidden_layer/gru_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/Hidden_layer/gru_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_input_layerPlaceholder*+
_output_shapes
:���������d*
dtype0* 
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerHidden_layer/gru_cell/biasHidden_layer/gru_cell/kernel&Hidden_layer/gru_cell/recurrent_kernelOutput_layer/kernelOutput_layer/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_10083362
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'Output_layer/kernel/Read/ReadVariableOp%Output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0Hidden_layer/gru_cell/kernel/Read/ReadVariableOp:Hidden_layer/gru_cell/recurrent_kernel/Read/ReadVariableOp.Hidden_layer/gru_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/Output_layer/kernel/m/Read/ReadVariableOp,Adam/Output_layer/bias/m/Read/ReadVariableOp7Adam/Hidden_layer/gru_cell/kernel/m/Read/ReadVariableOpAAdam/Hidden_layer/gru_cell/recurrent_kernel/m/Read/ReadVariableOp5Adam/Hidden_layer/gru_cell/bias/m/Read/ReadVariableOp.Adam/Output_layer/kernel/v/Read/ReadVariableOp,Adam/Output_layer/bias/v/Read/ReadVariableOp7Adam/Hidden_layer/gru_cell/kernel/v/Read/ReadVariableOpAAdam/Hidden_layer/gru_cell/recurrent_kernel/v/Read/ReadVariableOp5Adam/Hidden_layer/gru_cell/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_10084398
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameOutput_layer/kernelOutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateHidden_layer/gru_cell/kernel&Hidden_layer/gru_cell/recurrent_kernelHidden_layer/gru_cell/biastotalcounttotal_1count_1Adam/Output_layer/kernel/mAdam/Output_layer/bias/m#Adam/Hidden_layer/gru_cell/kernel/m-Adam/Hidden_layer/gru_cell/recurrent_kernel/m!Adam/Hidden_layer/gru_cell/bias/mAdam/Output_layer/kernel/vAdam/Output_layer/bias/v#Adam/Hidden_layer/gru_cell/kernel/v-Adam/Hidden_layer/gru_cell/recurrent_kernel/v!Adam/Hidden_layer/gru_cell/bias/v*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_10084480�
�
�
/__inference_Hidden_layer_layer_call_fn_10083406

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	d�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082825o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
while_cond_10083473
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10083473___redundant_placeholder06
2while_while_cond_10083473___redundant_placeholder16
2while_while_cond_10083473___redundant_placeholder26
2while_while_cond_10083473___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�4
�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082379

inputs$
gru_cell_10082303:	�$
gru_cell_10082305:	�$
gru_cell_10082307:	d�
identity�� gru_cell/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_10082303gru_cell_10082305gru_cell_10082307*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_10082263n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_10082303gru_cell_10082305gru_cell_10082307*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10082315*
condR
while_cond_10082314*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������dq
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
while_cond_10083658
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10083658___redundant_placeholder06
2while_while_cond_10083658___redundant_placeholder16
2while_while_cond_10083658___redundant_placeholder26
2while_while_cond_10083658___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�?
�
while_body_10083844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	d���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0~
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:c
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������dk
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������do
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������dg
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d~
while/gru_cell/mul_2Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������dY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_3Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_3AddV2while/gru_cell/mul_2:z:0while/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
/__inference_Hidden_layer_layer_call_fn_10083384
inputs_0
unknown:	�
	unknown_0:	�
	unknown_1:	d�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082379o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�6
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_10082263

inputs

states*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:���������O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:���������Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:�
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:���������Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:�
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:���������g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:���������u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������d[
add_2AddV2split:output:2	mul_1:z:0*
T0*'
_output_shapes
:���������dI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������dS
mul_2MulSigmoid:y:0states*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������d[
mul_3Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������dV
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�
�
while_body_10082105
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_10082127_0:	�,
while_gru_cell_10082129_0:	�,
while_gru_cell_10082131_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_10082127:	�*
while_gru_cell_10082129:	�*
while_gru_cell_10082131:	d���&while/gru_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_10082127_0while_gru_cell_10082129_0while_gru_cell_10082131_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_10082092�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������du

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_10082127while_gru_cell_10082127_0"4
while_gru_cell_10082129while_gru_cell_10082129_0"4
while_gru_cell_10082131while_gru_cell_10082131_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�y
�
#__inference__wrapped_model_10082018
input_layerK
8sequential_hidden_layer_gru_cell_readvariableop_resource:	�R
?sequential_hidden_layer_gru_cell_matmul_readvariableop_resource:	�T
Asequential_hidden_layer_gru_cell_matmul_1_readvariableop_resource:	d�H
6sequential_output_layer_matmul_readvariableop_resource:dE
7sequential_output_layer_biasadd_readvariableop_resource:
identity��6sequential/Hidden_layer/gru_cell/MatMul/ReadVariableOp�8sequential/Hidden_layer/gru_cell/MatMul_1/ReadVariableOp�/sequential/Hidden_layer/gru_cell/ReadVariableOp�sequential/Hidden_layer/while�.sequential/Output_layer/BiasAdd/ReadVariableOp�-sequential/Output_layer/MatMul/ReadVariableOpX
sequential/Hidden_layer/ShapeShapeinput_layer*
T0*
_output_shapes
:u
+sequential/Hidden_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential/Hidden_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential/Hidden_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential/Hidden_layer/strided_sliceStridedSlice&sequential/Hidden_layer/Shape:output:04sequential/Hidden_layer/strided_slice/stack:output:06sequential/Hidden_layer/strided_slice/stack_1:output:06sequential/Hidden_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential/Hidden_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
$sequential/Hidden_layer/zeros/packedPack.sequential/Hidden_layer/strided_slice:output:0/sequential/Hidden_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential/Hidden_layer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential/Hidden_layer/zerosFill-sequential/Hidden_layer/zeros/packed:output:0,sequential/Hidden_layer/zeros/Const:output:0*
T0*'
_output_shapes
:���������d{
&sequential/Hidden_layer/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
!sequential/Hidden_layer/transpose	Transposeinput_layer/sequential/Hidden_layer/transpose/perm:output:0*
T0*+
_output_shapes
:d���������t
sequential/Hidden_layer/Shape_1Shape%sequential/Hidden_layer/transpose:y:0*
T0*
_output_shapes
:w
-sequential/Hidden_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential/Hidden_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential/Hidden_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'sequential/Hidden_layer/strided_slice_1StridedSlice(sequential/Hidden_layer/Shape_1:output:06sequential/Hidden_layer/strided_slice_1/stack:output:08sequential/Hidden_layer/strided_slice_1/stack_1:output:08sequential/Hidden_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
3sequential/Hidden_layer/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
%sequential/Hidden_layer/TensorArrayV2TensorListReserve<sequential/Hidden_layer/TensorArrayV2/element_shape:output:00sequential/Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Msequential/Hidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
?sequential/Hidden_layer/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%sequential/Hidden_layer/transpose:y:0Vsequential/Hidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���w
-sequential/Hidden_layer/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential/Hidden_layer/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential/Hidden_layer/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'sequential/Hidden_layer/strided_slice_2StridedSlice%sequential/Hidden_layer/transpose:y:06sequential/Hidden_layer/strided_slice_2/stack:output:08sequential/Hidden_layer/strided_slice_2/stack_1:output:08sequential/Hidden_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
0sequential/Hidden_layer/gru_cell/ones_like/ShapeShape0sequential/Hidden_layer/strided_slice_2:output:0*
T0*
_output_shapes
:u
0sequential/Hidden_layer/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*sequential/Hidden_layer/gru_cell/ones_likeFill9sequential/Hidden_layer/gru_cell/ones_like/Shape:output:09sequential/Hidden_layer/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
/sequential/Hidden_layer/gru_cell/ReadVariableOpReadVariableOp8sequential_hidden_layer_gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0�
(sequential/Hidden_layer/gru_cell/unstackUnpack7sequential/Hidden_layer/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
$sequential/Hidden_layer/gru_cell/mulMul0sequential/Hidden_layer/strided_slice_2:output:03sequential/Hidden_layer/gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
6sequential/Hidden_layer/gru_cell/MatMul/ReadVariableOpReadVariableOp?sequential_hidden_layer_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'sequential/Hidden_layer/gru_cell/MatMulMatMul(sequential/Hidden_layer/gru_cell/mul:z:0>sequential/Hidden_layer/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(sequential/Hidden_layer/gru_cell/BiasAddBiasAdd1sequential/Hidden_layer/gru_cell/MatMul:product:01sequential/Hidden_layer/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������{
0sequential/Hidden_layer/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
&sequential/Hidden_layer/gru_cell/splitSplit9sequential/Hidden_layer/gru_cell/split/split_dim:output:01sequential/Hidden_layer/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
8sequential/Hidden_layer/gru_cell/MatMul_1/ReadVariableOpReadVariableOpAsequential_hidden_layer_gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
)sequential/Hidden_layer/gru_cell/MatMul_1MatMul&sequential/Hidden_layer/zeros:output:0@sequential/Hidden_layer/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*sequential/Hidden_layer/gru_cell/BiasAdd_1BiasAdd3sequential/Hidden_layer/gru_cell/MatMul_1:product:01sequential/Hidden_layer/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������{
&sequential/Hidden_layer/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����}
2sequential/Hidden_layer/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
(sequential/Hidden_layer/gru_cell/split_1SplitV3sequential/Hidden_layer/gru_cell/BiasAdd_1:output:0/sequential/Hidden_layer/gru_cell/Const:output:0;sequential/Hidden_layer/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
$sequential/Hidden_layer/gru_cell/addAddV2/sequential/Hidden_layer/gru_cell/split:output:01sequential/Hidden_layer/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d�
(sequential/Hidden_layer/gru_cell/SigmoidSigmoid(sequential/Hidden_layer/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
&sequential/Hidden_layer/gru_cell/add_1AddV2/sequential/Hidden_layer/gru_cell/split:output:11sequential/Hidden_layer/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������d�
*sequential/Hidden_layer/gru_cell/Sigmoid_1Sigmoid*sequential/Hidden_layer/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
&sequential/Hidden_layer/gru_cell/mul_1Mul.sequential/Hidden_layer/gru_cell/Sigmoid_1:y:01sequential/Hidden_layer/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
&sequential/Hidden_layer/gru_cell/add_2AddV2/sequential/Hidden_layer/gru_cell/split:output:2*sequential/Hidden_layer/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d�
%sequential/Hidden_layer/gru_cell/ReluRelu*sequential/Hidden_layer/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d�
&sequential/Hidden_layer/gru_cell/mul_2Mul,sequential/Hidden_layer/gru_cell/Sigmoid:y:0&sequential/Hidden_layer/zeros:output:0*
T0*'
_output_shapes
:���������dk
&sequential/Hidden_layer/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$sequential/Hidden_layer/gru_cell/subSub/sequential/Hidden_layer/gru_cell/sub/x:output:0,sequential/Hidden_layer/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
&sequential/Hidden_layer/gru_cell/mul_3Mul(sequential/Hidden_layer/gru_cell/sub:z:03sequential/Hidden_layer/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
&sequential/Hidden_layer/gru_cell/add_3AddV2*sequential/Hidden_layer/gru_cell/mul_2:z:0*sequential/Hidden_layer/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
5sequential/Hidden_layer/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
'sequential/Hidden_layer/TensorArrayV2_1TensorListReserve>sequential/Hidden_layer/TensorArrayV2_1/element_shape:output:00sequential/Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���^
sequential/Hidden_layer/timeConst*
_output_shapes
: *
dtype0*
value	B : {
0sequential/Hidden_layer/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������l
*sequential/Hidden_layer/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential/Hidden_layer/whileWhile3sequential/Hidden_layer/while/loop_counter:output:09sequential/Hidden_layer/while/maximum_iterations:output:0%sequential/Hidden_layer/time:output:00sequential/Hidden_layer/TensorArrayV2_1:handle:0&sequential/Hidden_layer/zeros:output:00sequential/Hidden_layer/strided_slice_1:output:0Osequential/Hidden_layer/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_hidden_layer_gru_cell_readvariableop_resource?sequential_hidden_layer_gru_cell_matmul_readvariableop_resourceAsequential_hidden_layer_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *7
body/R-
+sequential_Hidden_layer_while_body_10081919*7
cond/R-
+sequential_Hidden_layer_while_cond_10081918*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
Hsequential/Hidden_layer/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
:sequential/Hidden_layer/TensorArrayV2Stack/TensorListStackTensorListStack&sequential/Hidden_layer/while:output:3Qsequential/Hidden_layer/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d���������d*
element_dtype0�
-sequential/Hidden_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������y
/sequential/Hidden_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/sequential/Hidden_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'sequential/Hidden_layer/strided_slice_3StridedSliceCsequential/Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:06sequential/Hidden_layer/strided_slice_3/stack:output:08sequential/Hidden_layer/strided_slice_3/stack_1:output:08sequential/Hidden_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask}
(sequential/Hidden_layer/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
#sequential/Hidden_layer/transpose_1	TransposeCsequential/Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:01sequential/Hidden_layer/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dds
sequential/Hidden_layer/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
-sequential/Output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
sequential/Output_layer/MatMulMatMul0sequential/Hidden_layer/strided_slice_3:output:05sequential/Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential/Output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/Output_layer/BiasAddBiasAdd(sequential/Output_layer/MatMul:product:06sequential/Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(sequential/Output_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp7^sequential/Hidden_layer/gru_cell/MatMul/ReadVariableOp9^sequential/Hidden_layer/gru_cell/MatMul_1/ReadVariableOp0^sequential/Hidden_layer/gru_cell/ReadVariableOp^sequential/Hidden_layer/while/^sequential/Output_layer/BiasAdd/ReadVariableOp.^sequential/Output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 2p
6sequential/Hidden_layer/gru_cell/MatMul/ReadVariableOp6sequential/Hidden_layer/gru_cell/MatMul/ReadVariableOp2t
8sequential/Hidden_layer/gru_cell/MatMul_1/ReadVariableOp8sequential/Hidden_layer/gru_cell/MatMul_1/ReadVariableOp2b
/sequential/Hidden_layer/gru_cell/ReadVariableOp/sequential/Hidden_layer/gru_cell/ReadVariableOp2>
sequential/Hidden_layer/whilesequential/Hidden_layer/while2`
.sequential/Output_layer/BiasAdd/ReadVariableOp.sequential/Output_layer/BiasAdd/ReadVariableOp2^
-sequential/Output_layer/MatMul/ReadVariableOp-sequential/Output_layer/MatMul/ReadVariableOp:X T
+
_output_shapes
:���������d
%
_user_specified_nameinput_layer
�
�
/__inference_Hidden_layer_layer_call_fn_10083395

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	d�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_10083362
input_layer
unknown:	�
	unknown_0:	�
	unknown_1:	d�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_10082018o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������d
%
_user_specified_nameinput_layer
�
�
+sequential_Hidden_layer_while_cond_10081918L
Hsequential_hidden_layer_while_sequential_hidden_layer_while_loop_counterR
Nsequential_hidden_layer_while_sequential_hidden_layer_while_maximum_iterations-
)sequential_hidden_layer_while_placeholder/
+sequential_hidden_layer_while_placeholder_1/
+sequential_hidden_layer_while_placeholder_2N
Jsequential_hidden_layer_while_less_sequential_hidden_layer_strided_slice_1f
bsequential_hidden_layer_while_sequential_hidden_layer_while_cond_10081918___redundant_placeholder0f
bsequential_hidden_layer_while_sequential_hidden_layer_while_cond_10081918___redundant_placeholder1f
bsequential_hidden_layer_while_sequential_hidden_layer_while_cond_10081918___redundant_placeholder2f
bsequential_hidden_layer_while_sequential_hidden_layer_while_cond_10081918___redundant_placeholder3*
&sequential_hidden_layer_while_identity
�
"sequential/Hidden_layer/while/LessLess)sequential_hidden_layer_while_placeholderJsequential_hidden_layer_while_less_sequential_hidden_layer_strided_slice_1*
T0*
_output_shapes
: {
&sequential/Hidden_layer/while/IdentityIdentity&sequential/Hidden_layer/while/Less:z:0*
T0
*
_output_shapes
: "Y
&sequential_hidden_layer_while_identity/sequential/Hidden_layer/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�	
�
J__inference_Output_layer_layer_call_and_return_conditional_losses_10082573

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
while_cond_10082707
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10082707___redundant_placeholder06
2while_while_cond_10082707___redundant_placeholder16
2while_while_cond_10082707___redundant_placeholder26
2while_while_cond_10082707___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�h
�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10084146

inputs3
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	d�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask`
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:]
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������[
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*'
_output_shapes
:���������a
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:�
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*d
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:���������]
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������c
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:�
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2f
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������]
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:���������c
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:�
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2f
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:���������y
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num{
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:����������
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulgru_cell/mul:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������d}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������dc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������dz
gru_cell/mul_1Mulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������dv
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:���������dm
gru_cell/mul_2Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������dv
gru_cell/mul_3Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������dq
gru_cell/add_3AddV2gru_cell/mul_2:z:0gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10084029*
condR
while_cond_10084028*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d���������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������d: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
+__inference_gru_cell_layer_call_fn_10084179

inputs
states_0
unknown:	�
	unknown_0:	�
	unknown_1:	d�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_10082092o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�O
�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082555

inputs3
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	d�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask`
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:]
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������y
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num|
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulgru_cell/mul:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������d}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������dc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������dz
gru_cell/mul_1Mulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������dv
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:���������dm
gru_cell/mul_2Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������dv
gru_cell/mul_3Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������dq
gru_cell/add_3AddV2gru_cell/mul_2:z:0gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10082462*
condR
while_cond_10082461*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d���������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������d: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
while_cond_10082314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10082314___redundant_placeholder06
2while_while_cond_10082314___redundant_placeholder16
2while_while_cond_10082314___redundant_placeholder26
2while_while_cond_10082314___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�?
�
while_body_10083474
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	d���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0~
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:c
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������dk
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������do
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������dg
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d~
while/gru_cell/mul_2Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������dY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_3Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_3AddV2while/gru_cell/mul_2:z:0while/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_10083843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10083843___redundant_placeholder06
2while_while_cond_10083843___redundant_placeholder16
2while_while_cond_10083843___redundant_placeholder26
2while_while_cond_10083843___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�[
�
while_body_10083659
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	d���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0~
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:c
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������a
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*'
_output_shapes
:���������m
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*j
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:���������c
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������o
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2l
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������c
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:���������o
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2l
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:����������
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:����������
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������dk
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������do
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������dg
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d~
while/gru_cell/mul_2Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������dY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_3Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_3AddV2while/gru_cell/mul_2:z:0while/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�b
�
$__inference__traced_restore_10084480
file_prefix6
$assignvariableop_output_layer_kernel:d2
$assignvariableop_1_output_layer_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: B
/assignvariableop_7_hidden_layer_gru_cell_kernel:	�L
9assignvariableop_8_hidden_layer_gru_cell_recurrent_kernel:	d�@
-assignvariableop_9_hidden_layer_gru_cell_bias:	�#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: @
.assignvariableop_14_adam_output_layer_kernel_m:d:
,assignvariableop_15_adam_output_layer_bias_m:J
7assignvariableop_16_adam_hidden_layer_gru_cell_kernel_m:	�T
Aassignvariableop_17_adam_hidden_layer_gru_cell_recurrent_kernel_m:	d�H
5assignvariableop_18_adam_hidden_layer_gru_cell_bias_m:	�@
.assignvariableop_19_adam_output_layer_kernel_v:d:
,assignvariableop_20_adam_output_layer_bias_v:J
7assignvariableop_21_adam_hidden_layer_gru_cell_kernel_v:	�T
Aassignvariableop_22_adam_hidden_layer_gru_cell_recurrent_kernel_v:	d�H
5assignvariableop_23_adam_hidden_layer_gru_cell_bias_v:	�
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp$assignvariableop_output_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp$assignvariableop_1_output_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp/assignvariableop_7_hidden_layer_gru_cell_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp9assignvariableop_8_hidden_layer_gru_cell_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_hidden_layer_gru_cell_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_adam_output_layer_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_output_layer_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_adam_hidden_layer_gru_cell_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpAassignvariableop_17_adam_hidden_layer_gru_cell_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adam_hidden_layer_gru_cell_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_output_layer_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_output_layer_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_hidden_layer_gru_cell_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpAassignvariableop_22_adam_hidden_layer_gru_cell_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_hidden_layer_gru_cell_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�O
�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10083937

inputs3
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	d�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask`
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:]
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������y
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num|
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulgru_cell/mul:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������d}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������dc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������dz
gru_cell/mul_1Mulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������dv
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:���������dm
gru_cell/mul_2Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������dv
gru_cell/mul_3Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������dq
gru_cell/add_3AddV2gru_cell/mul_2:z:0gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10083844*
condR
while_cond_10083843*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d���������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������d: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_10082092

inputs

states*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:���������u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������d[
add_2AddV2split:output:2	mul_1:z:0*
T0*'
_output_shapes
:���������dI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������dS
mul_2MulSigmoid:y:0states*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������d[
mul_3Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������dV
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�
�
-__inference_sequential_layer_call_fn_10082593
input_layer
unknown:	�
	unknown_0:	�
	unknown_1:	d�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_10082580o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������d
%
_user_specified_nameinput_layer
�8
�
!__inference__traced_save_10084398
file_prefix2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_hidden_layer_gru_cell_kernel_read_readvariableopE
Asavev2_hidden_layer_gru_cell_recurrent_kernel_read_readvariableop9
5savev2_hidden_layer_gru_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableopB
>savev2_adam_hidden_layer_gru_cell_kernel_m_read_readvariableopL
Hsavev2_adam_hidden_layer_gru_cell_recurrent_kernel_m_read_readvariableop@
<savev2_adam_hidden_layer_gru_cell_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableopB
>savev2_adam_hidden_layer_gru_cell_kernel_v_read_readvariableopL
Hsavev2_adam_hidden_layer_gru_cell_recurrent_kernel_v_read_readvariableop@
<savev2_adam_hidden_layer_gru_cell_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_hidden_layer_gru_cell_kernel_read_readvariableopAsavev2_hidden_layer_gru_cell_recurrent_kernel_read_readvariableop5savev2_hidden_layer_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop>savev2_adam_hidden_layer_gru_cell_kernel_m_read_readvariableopHsavev2_adam_hidden_layer_gru_cell_recurrent_kernel_m_read_readvariableop<savev2_adam_hidden_layer_gru_cell_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableop>savev2_adam_hidden_layer_gru_cell_kernel_v_read_readvariableopHsavev2_adam_hidden_layer_gru_cell_recurrent_kernel_v_read_readvariableop<savev2_adam_hidden_layer_gru_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :d:: : : : : :	�:	d�:	�: : : : :d::	�:	d�:	�:d::	�:	d�:	�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:%	!

_output_shapes
:	d�:%
!

_output_shapes
:	�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	d�:%!

_output_shapes
:	�:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	d�:%!

_output_shapes
:	�:

_output_shapes
: 
�r
�

 Hidden_layer_while_body_100832226
2hidden_layer_while_hidden_layer_while_loop_counter<
8hidden_layer_while_hidden_layer_while_maximum_iterations"
hidden_layer_while_placeholder$
 hidden_layer_while_placeholder_1$
 hidden_layer_while_placeholder_25
1hidden_layer_while_hidden_layer_strided_slice_1_0q
mhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0H
5hidden_layer_while_gru_cell_readvariableop_resource_0:	�O
<hidden_layer_while_gru_cell_matmul_readvariableop_resource_0:	�Q
>hidden_layer_while_gru_cell_matmul_1_readvariableop_resource_0:	d�
hidden_layer_while_identity!
hidden_layer_while_identity_1!
hidden_layer_while_identity_2!
hidden_layer_while_identity_3!
hidden_layer_while_identity_43
/hidden_layer_while_hidden_layer_strided_slice_1o
khidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensorF
3hidden_layer_while_gru_cell_readvariableop_resource:	�M
:hidden_layer_while_gru_cell_matmul_readvariableop_resource:	�O
<hidden_layer_while_gru_cell_matmul_1_readvariableop_resource:	d���1Hidden_layer/while/gru_cell/MatMul/ReadVariableOp�3Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp�*Hidden_layer/while/gru_cell/ReadVariableOp�
DHidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6Hidden_layer/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0hidden_layer_while_placeholderMHidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+Hidden_layer/while/gru_cell/ones_like/ShapeShape=Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:p
+Hidden_layer/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Hidden_layer/while/gru_cell/ones_likeFill4Hidden_layer/while/gru_cell/ones_like/Shape:output:04Hidden_layer/while/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������n
)Hidden_layer/while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
'Hidden_layer/while/gru_cell/dropout/MulMul.Hidden_layer/while/gru_cell/ones_like:output:02Hidden_layer/while/gru_cell/dropout/Const:output:0*
T0*'
_output_shapes
:����������
)Hidden_layer/while/gru_cell/dropout/ShapeShape.Hidden_layer/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
@Hidden_layer/while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform2Hidden_layer/while/gru_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*w
2Hidden_layer/while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
0Hidden_layer/while/gru_cell/dropout/GreaterEqualGreaterEqualIHidden_layer/while/gru_cell/dropout/random_uniform/RandomUniform:output:0;Hidden_layer/while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
(Hidden_layer/while/gru_cell/dropout/CastCast4Hidden_layer/while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
)Hidden_layer/while/gru_cell/dropout/Mul_1Mul+Hidden_layer/while/gru_cell/dropout/Mul:z:0,Hidden_layer/while/gru_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:���������p
+Hidden_layer/while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
)Hidden_layer/while/gru_cell/dropout_1/MulMul.Hidden_layer/while/gru_cell/ones_like:output:04Hidden_layer/while/gru_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:����������
+Hidden_layer/while/gru_cell/dropout_1/ShapeShape.Hidden_layer/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
BHidden_layer/while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform4Hidden_layer/while/gru_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2y
4Hidden_layer/while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
2Hidden_layer/while/gru_cell/dropout_1/GreaterEqualGreaterEqualKHidden_layer/while/gru_cell/dropout_1/random_uniform/RandomUniform:output:0=Hidden_layer/while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
*Hidden_layer/while/gru_cell/dropout_1/CastCast6Hidden_layer/while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
+Hidden_layer/while/gru_cell/dropout_1/Mul_1Mul-Hidden_layer/while/gru_cell/dropout_1/Mul:z:0.Hidden_layer/while/gru_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������p
+Hidden_layer/while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
)Hidden_layer/while/gru_cell/dropout_2/MulMul.Hidden_layer/while/gru_cell/ones_like:output:04Hidden_layer/while/gru_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:����������
+Hidden_layer/while/gru_cell/dropout_2/ShapeShape.Hidden_layer/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
BHidden_layer/while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform4Hidden_layer/while/gru_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2y
4Hidden_layer/while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
2Hidden_layer/while/gru_cell/dropout_2/GreaterEqualGreaterEqualKHidden_layer/while/gru_cell/dropout_2/random_uniform/RandomUniform:output:0=Hidden_layer/while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
*Hidden_layer/while/gru_cell/dropout_2/CastCast6Hidden_layer/while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
+Hidden_layer/while/gru_cell/dropout_2/Mul_1Mul-Hidden_layer/while/gru_cell/dropout_2/Mul:z:0.Hidden_layer/while/gru_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:����������
*Hidden_layer/while/gru_cell/ReadVariableOpReadVariableOp5hidden_layer_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
#Hidden_layer/while/gru_cell/unstackUnpack2Hidden_layer/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
Hidden_layer/while/gru_cell/mulMul=Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0-Hidden_layer/while/gru_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:����������
1Hidden_layer/while/gru_cell/MatMul/ReadVariableOpReadVariableOp<hidden_layer_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
"Hidden_layer/while/gru_cell/MatMulMatMul#Hidden_layer/while/gru_cell/mul:z:09Hidden_layer/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#Hidden_layer/while/gru_cell/BiasAddBiasAdd,Hidden_layer/while/gru_cell/MatMul:product:0,Hidden_layer/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������v
+Hidden_layer/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
!Hidden_layer/while/gru_cell/splitSplit4Hidden_layer/while/gru_cell/split/split_dim:output:0,Hidden_layer/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
3Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp>hidden_layer_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
$Hidden_layer/while/gru_cell/MatMul_1MatMul hidden_layer_while_placeholder_2;Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%Hidden_layer/while/gru_cell/BiasAdd_1BiasAdd.Hidden_layer/while/gru_cell/MatMul_1:product:0,Hidden_layer/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������v
!Hidden_layer/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����x
-Hidden_layer/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
#Hidden_layer/while/gru_cell/split_1SplitV.Hidden_layer/while/gru_cell/BiasAdd_1:output:0*Hidden_layer/while/gru_cell/Const:output:06Hidden_layer/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
Hidden_layer/while/gru_cell/addAddV2*Hidden_layer/while/gru_cell/split:output:0,Hidden_layer/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d�
#Hidden_layer/while/gru_cell/SigmoidSigmoid#Hidden_layer/while/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/add_1AddV2*Hidden_layer/while/gru_cell/split:output:1,Hidden_layer/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������d�
%Hidden_layer/while/gru_cell/Sigmoid_1Sigmoid%Hidden_layer/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/mul_1Mul)Hidden_layer/while/gru_cell/Sigmoid_1:y:0,Hidden_layer/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/add_2AddV2*Hidden_layer/while/gru_cell/split:output:2%Hidden_layer/while/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d�
 Hidden_layer/while/gru_cell/ReluRelu%Hidden_layer/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/mul_2Mul'Hidden_layer/while/gru_cell/Sigmoid:y:0 hidden_layer_while_placeholder_2*
T0*'
_output_shapes
:���������df
!Hidden_layer/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Hidden_layer/while/gru_cell/subSub*Hidden_layer/while/gru_cell/sub/x:output:0'Hidden_layer/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/mul_3Mul#Hidden_layer/while/gru_cell/sub:z:0.Hidden_layer/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/add_3AddV2%Hidden_layer/while/gru_cell/mul_2:z:0%Hidden_layer/while/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
7Hidden_layer/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem hidden_layer_while_placeholder_1hidden_layer_while_placeholder%Hidden_layer/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���Z
Hidden_layer/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
Hidden_layer/while/addAddV2hidden_layer_while_placeholder!Hidden_layer/while/add/y:output:0*
T0*
_output_shapes
: \
Hidden_layer/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
Hidden_layer/while/add_1AddV22hidden_layer_while_hidden_layer_while_loop_counter#Hidden_layer/while/add_1/y:output:0*
T0*
_output_shapes
: �
Hidden_layer/while/IdentityIdentityHidden_layer/while/add_1:z:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: �
Hidden_layer/while/Identity_1Identity8hidden_layer_while_hidden_layer_while_maximum_iterations^Hidden_layer/while/NoOp*
T0*
_output_shapes
: �
Hidden_layer/while/Identity_2IdentityHidden_layer/while/add:z:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: �
Hidden_layer/while/Identity_3IdentityGHidden_layer/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: :����
Hidden_layer/while/Identity_4Identity%Hidden_layer/while/gru_cell/add_3:z:0^Hidden_layer/while/NoOp*
T0*'
_output_shapes
:���������d�
Hidden_layer/while/NoOpNoOp2^Hidden_layer/while/gru_cell/MatMul/ReadVariableOp4^Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp+^Hidden_layer/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "~
<hidden_layer_while_gru_cell_matmul_1_readvariableop_resource>hidden_layer_while_gru_cell_matmul_1_readvariableop_resource_0"z
:hidden_layer_while_gru_cell_matmul_readvariableop_resource<hidden_layer_while_gru_cell_matmul_readvariableop_resource_0"l
3hidden_layer_while_gru_cell_readvariableop_resource5hidden_layer_while_gru_cell_readvariableop_resource_0"d
/hidden_layer_while_hidden_layer_strided_slice_11hidden_layer_while_hidden_layer_strided_slice_1_0"C
hidden_layer_while_identity$Hidden_layer/while/Identity:output:0"G
hidden_layer_while_identity_1&Hidden_layer/while/Identity_1:output:0"G
hidden_layer_while_identity_2&Hidden_layer/while/Identity_2:output:0"G
hidden_layer_while_identity_3&Hidden_layer/while/Identity_3:output:0"G
hidden_layer_while_identity_4&Hidden_layer/while/Identity_4:output:0"�
khidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensormhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2f
1Hidden_layer/while/gru_cell/MatMul/ReadVariableOp1Hidden_layer/while/gru_cell/MatMul/ReadVariableOp2j
3Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp3Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp2X
*Hidden_layer/while/gru_cell/ReadVariableOp*Hidden_layer/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
/__inference_Hidden_layer_layer_call_fn_10083373
inputs_0
unknown:	�
	unknown_0:	�
	unknown_1:	d�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082169o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
� 
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_10084236

inputs
states_0*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:���������u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������d[
add_2AddV2split:output:2	mul_1:z:0*
T0*'
_output_shapes
:���������dI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������dU
mul_2MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������d[
mul_3Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������dV
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�
�
H__inference_sequential_layer_call_and_return_conditional_losses_10082580

inputs(
hidden_layer_10082556:	�(
hidden_layer_10082558:	�(
hidden_layer_10082560:	d�'
output_layer_10082574:d#
output_layer_10082576:
identity��$Hidden_layer/StatefulPartitionedCall�$Output_layer/StatefulPartitionedCall�
$Hidden_layer/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_10082556hidden_layer_10082558hidden_layer_10082560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082555�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall-Hidden_layer/StatefulPartitionedCall:output:0output_layer_10082574output_layer_10082576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_10082573|
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^Hidden_layer/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 2L
$Hidden_layer/StatefulPartitionedCall$Hidden_layer/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�O
�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10083567
inputs_03
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	d�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask`
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:]
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������y
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num|
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulgru_cell/mul:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������d}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������dc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������dz
gru_cell/mul_1Mulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������dv
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:���������dm
gru_cell/mul_2Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������dv
gru_cell/mul_3Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������dq
gru_cell/add_3AddV2gru_cell/mul_2:z:0gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10083474*
condR
while_cond_10083473*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�

�
+__inference_gru_cell_layer_call_fn_10084193

inputs
states_0
unknown:	�
	unknown_0:	�
	unknown_1:	d�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_10082263o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�
�
while_cond_10082104
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10082104___redundant_placeholder06
2while_while_cond_10082104___redundant_placeholder16
2while_while_cond_10082104___redundant_placeholder26
2while_while_cond_10082104___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
-__inference_sequential_layer_call_fn_10082895
input_layer
unknown:	�
	unknown_0:	�
	unknown_1:	d�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_10082867o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������d
%
_user_specified_nameinput_layer
�
�
while_body_10082315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_10082337_0:	�,
while_gru_cell_10082339_0:	�,
while_gru_cell_10082341_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_10082337:	�*
while_gru_cell_10082339:	�*
while_gru_cell_10082341:	d���&while/gru_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_10082337_0while_gru_cell_10082339_0while_gru_cell_10082341_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_10082263�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������du

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_10082337while_gru_cell_10082337_0"4
while_gru_cell_10082339while_gru_cell_10082339_0"4
while_gru_cell_10082341while_gru_cell_10082341_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�h
�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082825

inputs3
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	d�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask`
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:]
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������[
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*'
_output_shapes
:���������a
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:�
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*d
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:���������]
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������c
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:�
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2f
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������]
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:���������c
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:�
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2f
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:���������y
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num{
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:����������
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulgru_cell/mul:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������d}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������dc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������dz
gru_cell/mul_1Mulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������dv
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:���������dm
gru_cell/mul_2Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������dv
gru_cell/mul_3Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������dq
gru_cell/add_3AddV2gru_cell/mul_2:z:0gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10082708*
condR
while_cond_10082707*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d���������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������d: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�P
�

 Hidden_layer_while_body_100830316
2hidden_layer_while_hidden_layer_while_loop_counter<
8hidden_layer_while_hidden_layer_while_maximum_iterations"
hidden_layer_while_placeholder$
 hidden_layer_while_placeholder_1$
 hidden_layer_while_placeholder_25
1hidden_layer_while_hidden_layer_strided_slice_1_0q
mhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0H
5hidden_layer_while_gru_cell_readvariableop_resource_0:	�O
<hidden_layer_while_gru_cell_matmul_readvariableop_resource_0:	�Q
>hidden_layer_while_gru_cell_matmul_1_readvariableop_resource_0:	d�
hidden_layer_while_identity!
hidden_layer_while_identity_1!
hidden_layer_while_identity_2!
hidden_layer_while_identity_3!
hidden_layer_while_identity_43
/hidden_layer_while_hidden_layer_strided_slice_1o
khidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensorF
3hidden_layer_while_gru_cell_readvariableop_resource:	�M
:hidden_layer_while_gru_cell_matmul_readvariableop_resource:	�O
<hidden_layer_while_gru_cell_matmul_1_readvariableop_resource:	d���1Hidden_layer/while/gru_cell/MatMul/ReadVariableOp�3Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp�*Hidden_layer/while/gru_cell/ReadVariableOp�
DHidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6Hidden_layer/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0hidden_layer_while_placeholderMHidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+Hidden_layer/while/gru_cell/ones_like/ShapeShape=Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:p
+Hidden_layer/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Hidden_layer/while/gru_cell/ones_likeFill4Hidden_layer/while/gru_cell/ones_like/Shape:output:04Hidden_layer/while/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
*Hidden_layer/while/gru_cell/ReadVariableOpReadVariableOp5hidden_layer_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
#Hidden_layer/while/gru_cell/unstackUnpack2Hidden_layer/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
Hidden_layer/while/gru_cell/mulMul=Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0.Hidden_layer/while/gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
1Hidden_layer/while/gru_cell/MatMul/ReadVariableOpReadVariableOp<hidden_layer_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
"Hidden_layer/while/gru_cell/MatMulMatMul#Hidden_layer/while/gru_cell/mul:z:09Hidden_layer/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#Hidden_layer/while/gru_cell/BiasAddBiasAdd,Hidden_layer/while/gru_cell/MatMul:product:0,Hidden_layer/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������v
+Hidden_layer/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
!Hidden_layer/while/gru_cell/splitSplit4Hidden_layer/while/gru_cell/split/split_dim:output:0,Hidden_layer/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
3Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp>hidden_layer_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
$Hidden_layer/while/gru_cell/MatMul_1MatMul hidden_layer_while_placeholder_2;Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%Hidden_layer/while/gru_cell/BiasAdd_1BiasAdd.Hidden_layer/while/gru_cell/MatMul_1:product:0,Hidden_layer/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������v
!Hidden_layer/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����x
-Hidden_layer/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
#Hidden_layer/while/gru_cell/split_1SplitV.Hidden_layer/while/gru_cell/BiasAdd_1:output:0*Hidden_layer/while/gru_cell/Const:output:06Hidden_layer/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
Hidden_layer/while/gru_cell/addAddV2*Hidden_layer/while/gru_cell/split:output:0,Hidden_layer/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d�
#Hidden_layer/while/gru_cell/SigmoidSigmoid#Hidden_layer/while/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/add_1AddV2*Hidden_layer/while/gru_cell/split:output:1,Hidden_layer/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������d�
%Hidden_layer/while/gru_cell/Sigmoid_1Sigmoid%Hidden_layer/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/mul_1Mul)Hidden_layer/while/gru_cell/Sigmoid_1:y:0,Hidden_layer/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/add_2AddV2*Hidden_layer/while/gru_cell/split:output:2%Hidden_layer/while/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d�
 Hidden_layer/while/gru_cell/ReluRelu%Hidden_layer/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/mul_2Mul'Hidden_layer/while/gru_cell/Sigmoid:y:0 hidden_layer_while_placeholder_2*
T0*'
_output_shapes
:���������df
!Hidden_layer/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Hidden_layer/while/gru_cell/subSub*Hidden_layer/while/gru_cell/sub/x:output:0'Hidden_layer/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/mul_3Mul#Hidden_layer/while/gru_cell/sub:z:0.Hidden_layer/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
!Hidden_layer/while/gru_cell/add_3AddV2%Hidden_layer/while/gru_cell/mul_2:z:0%Hidden_layer/while/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
7Hidden_layer/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem hidden_layer_while_placeholder_1hidden_layer_while_placeholder%Hidden_layer/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���Z
Hidden_layer/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
Hidden_layer/while/addAddV2hidden_layer_while_placeholder!Hidden_layer/while/add/y:output:0*
T0*
_output_shapes
: \
Hidden_layer/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
Hidden_layer/while/add_1AddV22hidden_layer_while_hidden_layer_while_loop_counter#Hidden_layer/while/add_1/y:output:0*
T0*
_output_shapes
: �
Hidden_layer/while/IdentityIdentityHidden_layer/while/add_1:z:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: �
Hidden_layer/while/Identity_1Identity8hidden_layer_while_hidden_layer_while_maximum_iterations^Hidden_layer/while/NoOp*
T0*
_output_shapes
: �
Hidden_layer/while/Identity_2IdentityHidden_layer/while/add:z:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: �
Hidden_layer/while/Identity_3IdentityGHidden_layer/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: :����
Hidden_layer/while/Identity_4Identity%Hidden_layer/while/gru_cell/add_3:z:0^Hidden_layer/while/NoOp*
T0*'
_output_shapes
:���������d�
Hidden_layer/while/NoOpNoOp2^Hidden_layer/while/gru_cell/MatMul/ReadVariableOp4^Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp+^Hidden_layer/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "~
<hidden_layer_while_gru_cell_matmul_1_readvariableop_resource>hidden_layer_while_gru_cell_matmul_1_readvariableop_resource_0"z
:hidden_layer_while_gru_cell_matmul_readvariableop_resource<hidden_layer_while_gru_cell_matmul_readvariableop_resource_0"l
3hidden_layer_while_gru_cell_readvariableop_resource5hidden_layer_while_gru_cell_readvariableop_resource_0"d
/hidden_layer_while_hidden_layer_strided_slice_11hidden_layer_while_hidden_layer_strided_slice_1_0"C
hidden_layer_while_identity$Hidden_layer/while/Identity:output:0"G
hidden_layer_while_identity_1&Hidden_layer/while/Identity_1:output:0"G
hidden_layer_while_identity_2&Hidden_layer/while/Identity_2:output:0"G
hidden_layer_while_identity_3&Hidden_layer/while/Identity_3:output:0"G
hidden_layer_while_identity_4&Hidden_layer/while/Identity_4:output:0"�
khidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensormhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2f
1Hidden_layer/while/gru_cell/MatMul/ReadVariableOp1Hidden_layer/while/gru_cell/MatMul/ReadVariableOp2j
3Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp3Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp2X
*Hidden_layer/while/gru_cell/ReadVariableOp*Hidden_layer/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�

�
 Hidden_layer_while_cond_100832216
2hidden_layer_while_hidden_layer_while_loop_counter<
8hidden_layer_while_hidden_layer_while_maximum_iterations"
hidden_layer_while_placeholder$
 hidden_layer_while_placeholder_1$
 hidden_layer_while_placeholder_28
4hidden_layer_while_less_hidden_layer_strided_slice_1P
Lhidden_layer_while_hidden_layer_while_cond_10083221___redundant_placeholder0P
Lhidden_layer_while_hidden_layer_while_cond_10083221___redundant_placeholder1P
Lhidden_layer_while_hidden_layer_while_cond_10083221___redundant_placeholder2P
Lhidden_layer_while_hidden_layer_while_cond_10083221___redundant_placeholder3
hidden_layer_while_identity
�
Hidden_layer/while/LessLesshidden_layer_while_placeholder4hidden_layer_while_less_hidden_layer_strided_slice_1*
T0*
_output_shapes
: e
Hidden_layer/while/IdentityIdentityHidden_layer/while/Less:z:0*
T0
*
_output_shapes
: "C
hidden_layer_while_identity$Hidden_layer/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�

�
 Hidden_layer_while_cond_100830306
2hidden_layer_while_hidden_layer_while_loop_counter<
8hidden_layer_while_hidden_layer_while_maximum_iterations"
hidden_layer_while_placeholder$
 hidden_layer_while_placeholder_1$
 hidden_layer_while_placeholder_28
4hidden_layer_while_less_hidden_layer_strided_slice_1P
Lhidden_layer_while_hidden_layer_while_cond_10083030___redundant_placeholder0P
Lhidden_layer_while_hidden_layer_while_cond_10083030___redundant_placeholder1P
Lhidden_layer_while_hidden_layer_while_cond_10083030___redundant_placeholder2P
Lhidden_layer_while_hidden_layer_while_cond_10083030___redundant_placeholder3
hidden_layer_while_identity
�
Hidden_layer/while/LessLesshidden_layer_while_placeholder4hidden_layer_while_less_hidden_layer_strided_slice_1*
T0*
_output_shapes
: e
Hidden_layer/while/IdentityIdentityHidden_layer/while/Less:z:0*
T0
*
_output_shapes
: "C
hidden_layer_while_identity$Hidden_layer/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_sequential_layer_call_and_return_conditional_losses_10082911
input_layer(
hidden_layer_10082898:	�(
hidden_layer_10082900:	�(
hidden_layer_10082902:	d�'
output_layer_10082905:d#
output_layer_10082907:
identity��$Hidden_layer/StatefulPartitionedCall�$Output_layer/StatefulPartitionedCall�
$Hidden_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_10082898hidden_layer_10082900hidden_layer_10082902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082555�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall-Hidden_layer/StatefulPartitionedCall:output:0output_layer_10082905output_layer_10082907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_10082573|
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^Hidden_layer/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 2L
$Hidden_layer/StatefulPartitionedCall$Hidden_layer/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:X T
+
_output_shapes
:���������d
%
_user_specified_nameinput_layer
�4
�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082169

inputs$
gru_cell_10082093:	�$
gru_cell_10082095:	�$
gru_cell_10082097:	d�
identity�� gru_cell/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_10082093gru_cell_10082095gru_cell_10082097*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_gru_cell_layer_call_and_return_conditional_losses_10082092n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_10082093gru_cell_10082095gru_cell_10082097*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10082105*
condR
while_cond_10082104*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������dq
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�?
�
while_body_10082462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	d���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0~
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:c
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������dk
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������do
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������dg
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d~
while/gru_cell/mul_2Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������dY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_3Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_3AddV2while/gru_cell/mul_2:z:0while/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_sequential_layer_call_and_return_conditional_losses_10082927
input_layer(
hidden_layer_10082914:	�(
hidden_layer_10082916:	�(
hidden_layer_10082918:	d�'
output_layer_10082921:d#
output_layer_10082923:
identity��$Hidden_layer/StatefulPartitionedCall�$Output_layer/StatefulPartitionedCall�
$Hidden_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_10082914hidden_layer_10082916hidden_layer_10082918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082825�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall-Hidden_layer/StatefulPartitionedCall:output:0output_layer_10082921output_layer_10082923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_10082573|
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^Hidden_layer/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 2L
$Hidden_layer/StatefulPartitionedCall$Hidden_layer/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:X T
+
_output_shapes
:���������d
%
_user_specified_nameinput_layer
�6
�
F__inference_gru_cell_layer_call_and_return_conditional_losses_10084303

inputs
states_0*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:���������O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:���������Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:�
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:���������Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:�
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:���������g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:���������u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������d[
add_2AddV2split:output:2	mul_1:z:0*
T0*'
_output_shapes
:���������dI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������dU
mul_2MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������d[
mul_3Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������dV
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�
�
while_cond_10082461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10082461___redundant_placeholder06
2while_while_cond_10082461___redundant_placeholder16
2while_while_cond_10082461___redundant_placeholder26
2while_while_cond_10082461___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_sequential_layer_call_and_return_conditional_losses_10082867

inputs(
hidden_layer_10082854:	�(
hidden_layer_10082856:	�(
hidden_layer_10082858:	d�'
output_layer_10082861:d#
output_layer_10082863:
identity��$Hidden_layer/StatefulPartitionedCall�$Output_layer/StatefulPartitionedCall�
$Hidden_layer/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_10082854hidden_layer_10082856hidden_layer_10082858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10082825�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall-Hidden_layer/StatefulPartitionedCall:output:0output_layer_10082861output_layer_10082863*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_10082573|
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^Hidden_layer/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 2L
$Hidden_layer/StatefulPartitionedCall$Hidden_layer/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�h
�
H__inference_sequential_layer_call_and_return_conditional_losses_10083130

inputs@
-hidden_layer_gru_cell_readvariableop_resource:	�G
4hidden_layer_gru_cell_matmul_readvariableop_resource:	�I
6hidden_layer_gru_cell_matmul_1_readvariableop_resource:	d�=
+output_layer_matmul_readvariableop_resource:d:
,output_layer_biasadd_readvariableop_resource:
identity��+Hidden_layer/gru_cell/MatMul/ReadVariableOp�-Hidden_layer/gru_cell/MatMul_1/ReadVariableOp�$Hidden_layer/gru_cell/ReadVariableOp�Hidden_layer/while�#Output_layer/BiasAdd/ReadVariableOp�"Output_layer/MatMul/ReadVariableOpH
Hidden_layer/ShapeShapeinputs*
T0*
_output_shapes
:j
 Hidden_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Hidden_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Hidden_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hidden_layer/strided_sliceStridedSliceHidden_layer/Shape:output:0)Hidden_layer/strided_slice/stack:output:0+Hidden_layer/strided_slice/stack_1:output:0+Hidden_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
Hidden_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
Hidden_layer/zeros/packedPack#Hidden_layer/strided_slice:output:0$Hidden_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
Hidden_layer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Hidden_layer/zerosFill"Hidden_layer/zeros/packed:output:0!Hidden_layer/zeros/Const:output:0*
T0*'
_output_shapes
:���������dp
Hidden_layer/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
Hidden_layer/transpose	Transposeinputs$Hidden_layer/transpose/perm:output:0*
T0*+
_output_shapes
:d���������^
Hidden_layer/Shape_1ShapeHidden_layer/transpose:y:0*
T0*
_output_shapes
:l
"Hidden_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$Hidden_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$Hidden_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hidden_layer/strided_slice_1StridedSliceHidden_layer/Shape_1:output:0+Hidden_layer/strided_slice_1/stack:output:0-Hidden_layer/strided_slice_1/stack_1:output:0-Hidden_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(Hidden_layer/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Hidden_layer/TensorArrayV2TensorListReserve1Hidden_layer/TensorArrayV2/element_shape:output:0%Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
BHidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
4Hidden_layer/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorHidden_layer/transpose:y:0KHidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"Hidden_layer/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$Hidden_layer/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$Hidden_layer/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hidden_layer/strided_slice_2StridedSliceHidden_layer/transpose:y:0+Hidden_layer/strided_slice_2/stack:output:0-Hidden_layer/strided_slice_2/stack_1:output:0-Hidden_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskz
%Hidden_layer/gru_cell/ones_like/ShapeShape%Hidden_layer/strided_slice_2:output:0*
T0*
_output_shapes
:j
%Hidden_layer/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Hidden_layer/gru_cell/ones_likeFill.Hidden_layer/gru_cell/ones_like/Shape:output:0.Hidden_layer/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
$Hidden_layer/gru_cell/ReadVariableOpReadVariableOp-hidden_layer_gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Hidden_layer/gru_cell/unstackUnpack,Hidden_layer/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
Hidden_layer/gru_cell/mulMul%Hidden_layer/strided_slice_2:output:0(Hidden_layer/gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
+Hidden_layer/gru_cell/MatMul/ReadVariableOpReadVariableOp4hidden_layer_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Hidden_layer/gru_cell/MatMulMatMulHidden_layer/gru_cell/mul:z:03Hidden_layer/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Hidden_layer/gru_cell/BiasAddBiasAdd&Hidden_layer/gru_cell/MatMul:product:0&Hidden_layer/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������p
%Hidden_layer/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Hidden_layer/gru_cell/splitSplit.Hidden_layer/gru_cell/split/split_dim:output:0&Hidden_layer/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
-Hidden_layer/gru_cell/MatMul_1/ReadVariableOpReadVariableOp6hidden_layer_gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
Hidden_layer/gru_cell/MatMul_1MatMulHidden_layer/zeros:output:05Hidden_layer/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Hidden_layer/gru_cell/BiasAdd_1BiasAdd(Hidden_layer/gru_cell/MatMul_1:product:0&Hidden_layer/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������p
Hidden_layer/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����r
'Hidden_layer/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Hidden_layer/gru_cell/split_1SplitV(Hidden_layer/gru_cell/BiasAdd_1:output:0$Hidden_layer/gru_cell/Const:output:00Hidden_layer/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
Hidden_layer/gru_cell/addAddV2$Hidden_layer/gru_cell/split:output:0&Hidden_layer/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������dy
Hidden_layer/gru_cell/SigmoidSigmoidHidden_layer/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/add_1AddV2$Hidden_layer/gru_cell/split:output:1&Hidden_layer/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������d}
Hidden_layer/gru_cell/Sigmoid_1SigmoidHidden_layer/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/mul_1Mul#Hidden_layer/gru_cell/Sigmoid_1:y:0&Hidden_layer/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/add_2AddV2$Hidden_layer/gru_cell/split:output:2Hidden_layer/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������du
Hidden_layer/gru_cell/ReluReluHidden_layer/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/mul_2Mul!Hidden_layer/gru_cell/Sigmoid:y:0Hidden_layer/zeros:output:0*
T0*'
_output_shapes
:���������d`
Hidden_layer/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Hidden_layer/gru_cell/subSub$Hidden_layer/gru_cell/sub/x:output:0!Hidden_layer/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/mul_3MulHidden_layer/gru_cell/sub:z:0(Hidden_layer/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/add_3AddV2Hidden_layer/gru_cell/mul_2:z:0Hidden_layer/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d{
*Hidden_layer/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
Hidden_layer/TensorArrayV2_1TensorListReserve3Hidden_layer/TensorArrayV2_1/element_shape:output:0%Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
Hidden_layer/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%Hidden_layer/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
Hidden_layer/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
Hidden_layer/whileWhile(Hidden_layer/while/loop_counter:output:0.Hidden_layer/while/maximum_iterations:output:0Hidden_layer/time:output:0%Hidden_layer/TensorArrayV2_1:handle:0Hidden_layer/zeros:output:0%Hidden_layer/strided_slice_1:output:0DHidden_layer/TensorArrayUnstack/TensorListFromTensor:output_handle:0-hidden_layer_gru_cell_readvariableop_resource4hidden_layer_gru_cell_matmul_readvariableop_resource6hidden_layer_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 Hidden_layer_while_body_10083031*,
cond$R"
 Hidden_layer_while_cond_10083030*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
=Hidden_layer/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
/Hidden_layer/TensorArrayV2Stack/TensorListStackTensorListStackHidden_layer/while:output:3FHidden_layer/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d���������d*
element_dtype0u
"Hidden_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$Hidden_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$Hidden_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hidden_layer/strided_slice_3StridedSlice8Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:0+Hidden_layer/strided_slice_3/stack:output:0-Hidden_layer/strided_slice_3/stack_1:output:0-Hidden_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskr
Hidden_layer/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
Hidden_layer/transpose_1	Transpose8Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:0&Hidden_layer/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������ddh
Hidden_layer/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
"Output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
Output_layer/MatMulMatMul%Hidden_layer/strided_slice_3:output:0*Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output_layer/BiasAddBiasAddOutput_layer/MatMul:product:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
IdentityIdentityOutput_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^Hidden_layer/gru_cell/MatMul/ReadVariableOp.^Hidden_layer/gru_cell/MatMul_1/ReadVariableOp%^Hidden_layer/gru_cell/ReadVariableOp^Hidden_layer/while$^Output_layer/BiasAdd/ReadVariableOp#^Output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 2Z
+Hidden_layer/gru_cell/MatMul/ReadVariableOp+Hidden_layer/gru_cell/MatMul/ReadVariableOp2^
-Hidden_layer/gru_cell/MatMul_1/ReadVariableOp-Hidden_layer/gru_cell/MatMul_1/ReadVariableOp2L
$Hidden_layer/gru_cell/ReadVariableOp$Hidden_layer/gru_cell/ReadVariableOp2(
Hidden_layer/whileHidden_layer/while2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2H
"Output_layer/MatMul/ReadVariableOp"Output_layer/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
J__inference_Output_layer_layer_call_and_return_conditional_losses_10084165

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�[
�
while_body_10084029
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	d���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0~
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:c
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������a
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*'
_output_shapes
:���������m
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*j
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:���������c
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������o
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2l
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������c
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:���������o
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2l
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:����������
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:����������
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������dk
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������do
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������dg
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d~
while/gru_cell/mul_2Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������dY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_3Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_3AddV2while/gru_cell/mul_2:z:0while/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_sequential_layer_call_and_return_conditional_losses_10083345

inputs@
-hidden_layer_gru_cell_readvariableop_resource:	�G
4hidden_layer_gru_cell_matmul_readvariableop_resource:	�I
6hidden_layer_gru_cell_matmul_1_readvariableop_resource:	d�=
+output_layer_matmul_readvariableop_resource:d:
,output_layer_biasadd_readvariableop_resource:
identity��+Hidden_layer/gru_cell/MatMul/ReadVariableOp�-Hidden_layer/gru_cell/MatMul_1/ReadVariableOp�$Hidden_layer/gru_cell/ReadVariableOp�Hidden_layer/while�#Output_layer/BiasAdd/ReadVariableOp�"Output_layer/MatMul/ReadVariableOpH
Hidden_layer/ShapeShapeinputs*
T0*
_output_shapes
:j
 Hidden_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Hidden_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Hidden_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hidden_layer/strided_sliceStridedSliceHidden_layer/Shape:output:0)Hidden_layer/strided_slice/stack:output:0+Hidden_layer/strided_slice/stack_1:output:0+Hidden_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
Hidden_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
Hidden_layer/zeros/packedPack#Hidden_layer/strided_slice:output:0$Hidden_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
Hidden_layer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Hidden_layer/zerosFill"Hidden_layer/zeros/packed:output:0!Hidden_layer/zeros/Const:output:0*
T0*'
_output_shapes
:���������dp
Hidden_layer/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
Hidden_layer/transpose	Transposeinputs$Hidden_layer/transpose/perm:output:0*
T0*+
_output_shapes
:d���������^
Hidden_layer/Shape_1ShapeHidden_layer/transpose:y:0*
T0*
_output_shapes
:l
"Hidden_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$Hidden_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$Hidden_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hidden_layer/strided_slice_1StridedSliceHidden_layer/Shape_1:output:0+Hidden_layer/strided_slice_1/stack:output:0-Hidden_layer/strided_slice_1/stack_1:output:0-Hidden_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(Hidden_layer/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Hidden_layer/TensorArrayV2TensorListReserve1Hidden_layer/TensorArrayV2/element_shape:output:0%Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
BHidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
4Hidden_layer/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorHidden_layer/transpose:y:0KHidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"Hidden_layer/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$Hidden_layer/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$Hidden_layer/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hidden_layer/strided_slice_2StridedSliceHidden_layer/transpose:y:0+Hidden_layer/strided_slice_2/stack:output:0-Hidden_layer/strided_slice_2/stack_1:output:0-Hidden_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskz
%Hidden_layer/gru_cell/ones_like/ShapeShape%Hidden_layer/strided_slice_2:output:0*
T0*
_output_shapes
:j
%Hidden_layer/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Hidden_layer/gru_cell/ones_likeFill.Hidden_layer/gru_cell/ones_like/Shape:output:0.Hidden_layer/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������h
#Hidden_layer/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
!Hidden_layer/gru_cell/dropout/MulMul(Hidden_layer/gru_cell/ones_like:output:0,Hidden_layer/gru_cell/dropout/Const:output:0*
T0*'
_output_shapes
:���������{
#Hidden_layer/gru_cell/dropout/ShapeShape(Hidden_layer/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
:Hidden_layer/gru_cell/dropout/random_uniform/RandomUniformRandomUniform,Hidden_layer/gru_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*q
,Hidden_layer/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
*Hidden_layer/gru_cell/dropout/GreaterEqualGreaterEqualCHidden_layer/gru_cell/dropout/random_uniform/RandomUniform:output:05Hidden_layer/gru_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
"Hidden_layer/gru_cell/dropout/CastCast.Hidden_layer/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
#Hidden_layer/gru_cell/dropout/Mul_1Mul%Hidden_layer/gru_cell/dropout/Mul:z:0&Hidden_layer/gru_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:���������j
%Hidden_layer/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
#Hidden_layer/gru_cell/dropout_1/MulMul(Hidden_layer/gru_cell/ones_like:output:0.Hidden_layer/gru_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������}
%Hidden_layer/gru_cell/dropout_1/ShapeShape(Hidden_layer/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
<Hidden_layer/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform.Hidden_layer/gru_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2s
.Hidden_layer/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
,Hidden_layer/gru_cell/dropout_1/GreaterEqualGreaterEqualEHidden_layer/gru_cell/dropout_1/random_uniform/RandomUniform:output:07Hidden_layer/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
$Hidden_layer/gru_cell/dropout_1/CastCast0Hidden_layer/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
%Hidden_layer/gru_cell/dropout_1/Mul_1Mul'Hidden_layer/gru_cell/dropout_1/Mul:z:0(Hidden_layer/gru_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������j
%Hidden_layer/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
#Hidden_layer/gru_cell/dropout_2/MulMul(Hidden_layer/gru_cell/ones_like:output:0.Hidden_layer/gru_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:���������}
%Hidden_layer/gru_cell/dropout_2/ShapeShape(Hidden_layer/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
<Hidden_layer/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform.Hidden_layer/gru_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2s
.Hidden_layer/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
,Hidden_layer/gru_cell/dropout_2/GreaterEqualGreaterEqualEHidden_layer/gru_cell/dropout_2/random_uniform/RandomUniform:output:07Hidden_layer/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
$Hidden_layer/gru_cell/dropout_2/CastCast0Hidden_layer/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
%Hidden_layer/gru_cell/dropout_2/Mul_1Mul'Hidden_layer/gru_cell/dropout_2/Mul:z:0(Hidden_layer/gru_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:����������
$Hidden_layer/gru_cell/ReadVariableOpReadVariableOp-hidden_layer_gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Hidden_layer/gru_cell/unstackUnpack,Hidden_layer/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
Hidden_layer/gru_cell/mulMul%Hidden_layer/strided_slice_2:output:0'Hidden_layer/gru_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:����������
+Hidden_layer/gru_cell/MatMul/ReadVariableOpReadVariableOp4hidden_layer_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Hidden_layer/gru_cell/MatMulMatMulHidden_layer/gru_cell/mul:z:03Hidden_layer/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Hidden_layer/gru_cell/BiasAddBiasAdd&Hidden_layer/gru_cell/MatMul:product:0&Hidden_layer/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������p
%Hidden_layer/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Hidden_layer/gru_cell/splitSplit.Hidden_layer/gru_cell/split/split_dim:output:0&Hidden_layer/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
-Hidden_layer/gru_cell/MatMul_1/ReadVariableOpReadVariableOp6hidden_layer_gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
Hidden_layer/gru_cell/MatMul_1MatMulHidden_layer/zeros:output:05Hidden_layer/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Hidden_layer/gru_cell/BiasAdd_1BiasAdd(Hidden_layer/gru_cell/MatMul_1:product:0&Hidden_layer/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������p
Hidden_layer/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����r
'Hidden_layer/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Hidden_layer/gru_cell/split_1SplitV(Hidden_layer/gru_cell/BiasAdd_1:output:0$Hidden_layer/gru_cell/Const:output:00Hidden_layer/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
Hidden_layer/gru_cell/addAddV2$Hidden_layer/gru_cell/split:output:0&Hidden_layer/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������dy
Hidden_layer/gru_cell/SigmoidSigmoidHidden_layer/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/add_1AddV2$Hidden_layer/gru_cell/split:output:1&Hidden_layer/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������d}
Hidden_layer/gru_cell/Sigmoid_1SigmoidHidden_layer/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/mul_1Mul#Hidden_layer/gru_cell/Sigmoid_1:y:0&Hidden_layer/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/add_2AddV2$Hidden_layer/gru_cell/split:output:2Hidden_layer/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������du
Hidden_layer/gru_cell/ReluReluHidden_layer/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/mul_2Mul!Hidden_layer/gru_cell/Sigmoid:y:0Hidden_layer/zeros:output:0*
T0*'
_output_shapes
:���������d`
Hidden_layer/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Hidden_layer/gru_cell/subSub$Hidden_layer/gru_cell/sub/x:output:0!Hidden_layer/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/mul_3MulHidden_layer/gru_cell/sub:z:0(Hidden_layer/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
Hidden_layer/gru_cell/add_3AddV2Hidden_layer/gru_cell/mul_2:z:0Hidden_layer/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d{
*Hidden_layer/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
Hidden_layer/TensorArrayV2_1TensorListReserve3Hidden_layer/TensorArrayV2_1/element_shape:output:0%Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
Hidden_layer/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%Hidden_layer/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
Hidden_layer/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
Hidden_layer/whileWhile(Hidden_layer/while/loop_counter:output:0.Hidden_layer/while/maximum_iterations:output:0Hidden_layer/time:output:0%Hidden_layer/TensorArrayV2_1:handle:0Hidden_layer/zeros:output:0%Hidden_layer/strided_slice_1:output:0DHidden_layer/TensorArrayUnstack/TensorListFromTensor:output_handle:0-hidden_layer_gru_cell_readvariableop_resource4hidden_layer_gru_cell_matmul_readvariableop_resource6hidden_layer_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 Hidden_layer_while_body_10083222*,
cond$R"
 Hidden_layer_while_cond_10083221*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
=Hidden_layer/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
/Hidden_layer/TensorArrayV2Stack/TensorListStackTensorListStackHidden_layer/while:output:3FHidden_layer/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d���������d*
element_dtype0u
"Hidden_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$Hidden_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$Hidden_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Hidden_layer/strided_slice_3StridedSlice8Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:0+Hidden_layer/strided_slice_3/stack:output:0-Hidden_layer/strided_slice_3/stack_1:output:0-Hidden_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskr
Hidden_layer/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
Hidden_layer/transpose_1	Transpose8Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:0&Hidden_layer/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������ddh
Hidden_layer/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
"Output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
Output_layer/MatMulMatMul%Hidden_layer/strided_slice_3:output:0*Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output_layer/BiasAddBiasAddOutput_layer/MatMul:product:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
IdentityIdentityOutput_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^Hidden_layer/gru_cell/MatMul/ReadVariableOp.^Hidden_layer/gru_cell/MatMul_1/ReadVariableOp%^Hidden_layer/gru_cell/ReadVariableOp^Hidden_layer/while$^Output_layer/BiasAdd/ReadVariableOp#^Output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 2Z
+Hidden_layer/gru_cell/MatMul/ReadVariableOp+Hidden_layer/gru_cell/MatMul/ReadVariableOp2^
-Hidden_layer/gru_cell/MatMul_1/ReadVariableOp-Hidden_layer/gru_cell/MatMul_1/ReadVariableOp2L
$Hidden_layer/gru_cell/ReadVariableOp$Hidden_layer/gru_cell/ReadVariableOp2(
Hidden_layer/whileHidden_layer/while2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2H
"Output_layer/MatMul/ReadVariableOp"Output_layer/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
-__inference_sequential_layer_call_fn_10082963

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	d�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_10082867o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�[
�
while_body_10082708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	d�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	d���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0~
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:c
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������a
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*'
_output_shapes
:���������m
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*j
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:���������c
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������o
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2l
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������c
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:���������o
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:�
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2l
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:����������
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:����������
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������dk
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������do
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������dg
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d~
while/gru_cell/mul_2Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������dY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/mul_3Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/gru_cell/add_3AddV2while/gru_cell/mul_2:z:0while/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�i
�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10083776
inputs_03
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	d�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask`
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:]
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:���������[
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*'
_output_shapes
:���������a
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:�
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed*d
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:���������]
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������c
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:�
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2f
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������]
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:���������c
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:�
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2f
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:���������y
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num{
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:����������
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulgru_cell/mul:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������d}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������dc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������dz
gru_cell/mul_1Mulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������dv
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:���������dm
gru_cell/mul_2Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������dv
gru_cell/mul_3Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������dq
gru_cell/add_3AddV2gru_cell/mul_2:z:0gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10083659*
condR
while_cond_10083658*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
/__inference_Output_layer_layer_call_fn_10084155

inputs
unknown:d
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_10082573o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
-__inference_sequential_layer_call_fn_10082948

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	d�
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_10082580o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�_
�
+sequential_Hidden_layer_while_body_10081919L
Hsequential_hidden_layer_while_sequential_hidden_layer_while_loop_counterR
Nsequential_hidden_layer_while_sequential_hidden_layer_while_maximum_iterations-
)sequential_hidden_layer_while_placeholder/
+sequential_hidden_layer_while_placeholder_1/
+sequential_hidden_layer_while_placeholder_2K
Gsequential_hidden_layer_while_sequential_hidden_layer_strided_slice_1_0�
�sequential_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0S
@sequential_hidden_layer_while_gru_cell_readvariableop_resource_0:	�Z
Gsequential_hidden_layer_while_gru_cell_matmul_readvariableop_resource_0:	�\
Isequential_hidden_layer_while_gru_cell_matmul_1_readvariableop_resource_0:	d�*
&sequential_hidden_layer_while_identity,
(sequential_hidden_layer_while_identity_1,
(sequential_hidden_layer_while_identity_2,
(sequential_hidden_layer_while_identity_3,
(sequential_hidden_layer_while_identity_4I
Esequential_hidden_layer_while_sequential_hidden_layer_strided_slice_1�
�sequential_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_hidden_layer_tensorarrayunstack_tensorlistfromtensorQ
>sequential_hidden_layer_while_gru_cell_readvariableop_resource:	�X
Esequential_hidden_layer_while_gru_cell_matmul_readvariableop_resource:	�Z
Gsequential_hidden_layer_while_gru_cell_matmul_1_readvariableop_resource:	d���<sequential/Hidden_layer/while/gru_cell/MatMul/ReadVariableOp�>sequential/Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp�5sequential/Hidden_layer/while/gru_cell/ReadVariableOp�
Osequential/Hidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Asequential/Hidden_layer/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0)sequential_hidden_layer_while_placeholderXsequential/Hidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
6sequential/Hidden_layer/while/gru_cell/ones_like/ShapeShapeHsequential/Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:{
6sequential/Hidden_layer/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
0sequential/Hidden_layer/while/gru_cell/ones_likeFill?sequential/Hidden_layer/while/gru_cell/ones_like/Shape:output:0?sequential/Hidden_layer/while/gru_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
5sequential/Hidden_layer/while/gru_cell/ReadVariableOpReadVariableOp@sequential_hidden_layer_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
.sequential/Hidden_layer/while/gru_cell/unstackUnpack=sequential/Hidden_layer/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
*sequential/Hidden_layer/while/gru_cell/mulMulHsequential/Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:09sequential/Hidden_layer/while/gru_cell/ones_like:output:0*
T0*'
_output_shapes
:����������
<sequential/Hidden_layer/while/gru_cell/MatMul/ReadVariableOpReadVariableOpGsequential_hidden_layer_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
-sequential/Hidden_layer/while/gru_cell/MatMulMatMul.sequential/Hidden_layer/while/gru_cell/mul:z:0Dsequential/Hidden_layer/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential/Hidden_layer/while/gru_cell/BiasAddBiasAdd7sequential/Hidden_layer/while/gru_cell/MatMul:product:07sequential/Hidden_layer/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:�����������
6sequential/Hidden_layer/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
,sequential/Hidden_layer/while/gru_cell/splitSplit?sequential/Hidden_layer/while/gru_cell/split/split_dim:output:07sequential/Hidden_layer/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
>sequential/Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpIsequential_hidden_layer_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
/sequential/Hidden_layer/while/gru_cell/MatMul_1MatMul+sequential_hidden_layer_while_placeholder_2Fsequential/Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0sequential/Hidden_layer/while/gru_cell/BiasAdd_1BiasAdd9sequential/Hidden_layer/while/gru_cell/MatMul_1:product:07sequential/Hidden_layer/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:�����������
,sequential/Hidden_layer/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   �����
8sequential/Hidden_layer/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
.sequential/Hidden_layer/while/gru_cell/split_1SplitV9sequential/Hidden_layer/while/gru_cell/BiasAdd_1:output:05sequential/Hidden_layer/while/gru_cell/Const:output:0Asequential/Hidden_layer/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������d:���������d:���������d*
	num_split�
*sequential/Hidden_layer/while/gru_cell/addAddV25sequential/Hidden_layer/while/gru_cell/split:output:07sequential/Hidden_layer/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������d�
.sequential/Hidden_layer/while/gru_cell/SigmoidSigmoid.sequential/Hidden_layer/while/gru_cell/add:z:0*
T0*'
_output_shapes
:���������d�
,sequential/Hidden_layer/while/gru_cell/add_1AddV25sequential/Hidden_layer/while/gru_cell/split:output:17sequential/Hidden_layer/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������d�
0sequential/Hidden_layer/while/gru_cell/Sigmoid_1Sigmoid0sequential/Hidden_layer/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������d�
,sequential/Hidden_layer/while/gru_cell/mul_1Mul4sequential/Hidden_layer/while/gru_cell/Sigmoid_1:y:07sequential/Hidden_layer/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������d�
,sequential/Hidden_layer/while/gru_cell/add_2AddV25sequential/Hidden_layer/while/gru_cell/split:output:20sequential/Hidden_layer/while/gru_cell/mul_1:z:0*
T0*'
_output_shapes
:���������d�
+sequential/Hidden_layer/while/gru_cell/ReluRelu0sequential/Hidden_layer/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������d�
,sequential/Hidden_layer/while/gru_cell/mul_2Mul2sequential/Hidden_layer/while/gru_cell/Sigmoid:y:0+sequential_hidden_layer_while_placeholder_2*
T0*'
_output_shapes
:���������dq
,sequential/Hidden_layer/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*sequential/Hidden_layer/while/gru_cell/subSub5sequential/Hidden_layer/while/gru_cell/sub/x:output:02sequential/Hidden_layer/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������d�
,sequential/Hidden_layer/while/gru_cell/mul_3Mul.sequential/Hidden_layer/while/gru_cell/sub:z:09sequential/Hidden_layer/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:���������d�
,sequential/Hidden_layer/while/gru_cell/add_3AddV20sequential/Hidden_layer/while/gru_cell/mul_2:z:00sequential/Hidden_layer/while/gru_cell/mul_3:z:0*
T0*'
_output_shapes
:���������d�
Bsequential/Hidden_layer/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+sequential_hidden_layer_while_placeholder_1)sequential_hidden_layer_while_placeholder0sequential/Hidden_layer/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���e
#sequential/Hidden_layer/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
!sequential/Hidden_layer/while/addAddV2)sequential_hidden_layer_while_placeholder,sequential/Hidden_layer/while/add/y:output:0*
T0*
_output_shapes
: g
%sequential/Hidden_layer/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
#sequential/Hidden_layer/while/add_1AddV2Hsequential_hidden_layer_while_sequential_hidden_layer_while_loop_counter.sequential/Hidden_layer/while/add_1/y:output:0*
T0*
_output_shapes
: �
&sequential/Hidden_layer/while/IdentityIdentity'sequential/Hidden_layer/while/add_1:z:0#^sequential/Hidden_layer/while/NoOp*
T0*
_output_shapes
: �
(sequential/Hidden_layer/while/Identity_1IdentityNsequential_hidden_layer_while_sequential_hidden_layer_while_maximum_iterations#^sequential/Hidden_layer/while/NoOp*
T0*
_output_shapes
: �
(sequential/Hidden_layer/while/Identity_2Identity%sequential/Hidden_layer/while/add:z:0#^sequential/Hidden_layer/while/NoOp*
T0*
_output_shapes
: �
(sequential/Hidden_layer/while/Identity_3IdentityRsequential/Hidden_layer/while/TensorArrayV2Write/TensorListSetItem:output_handle:0#^sequential/Hidden_layer/while/NoOp*
T0*
_output_shapes
: :����
(sequential/Hidden_layer/while/Identity_4Identity0sequential/Hidden_layer/while/gru_cell/add_3:z:0#^sequential/Hidden_layer/while/NoOp*
T0*'
_output_shapes
:���������d�
"sequential/Hidden_layer/while/NoOpNoOp=^sequential/Hidden_layer/while/gru_cell/MatMul/ReadVariableOp?^sequential/Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp6^sequential/Hidden_layer/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "�
Gsequential_hidden_layer_while_gru_cell_matmul_1_readvariableop_resourceIsequential_hidden_layer_while_gru_cell_matmul_1_readvariableop_resource_0"�
Esequential_hidden_layer_while_gru_cell_matmul_readvariableop_resourceGsequential_hidden_layer_while_gru_cell_matmul_readvariableop_resource_0"�
>sequential_hidden_layer_while_gru_cell_readvariableop_resource@sequential_hidden_layer_while_gru_cell_readvariableop_resource_0"Y
&sequential_hidden_layer_while_identity/sequential/Hidden_layer/while/Identity:output:0"]
(sequential_hidden_layer_while_identity_11sequential/Hidden_layer/while/Identity_1:output:0"]
(sequential_hidden_layer_while_identity_21sequential/Hidden_layer/while/Identity_2:output:0"]
(sequential_hidden_layer_while_identity_31sequential/Hidden_layer/while/Identity_3:output:0"]
(sequential_hidden_layer_while_identity_41sequential/Hidden_layer/while/Identity_4:output:0"�
Esequential_hidden_layer_while_sequential_hidden_layer_strided_slice_1Gsequential_hidden_layer_while_sequential_hidden_layer_strided_slice_1_0"�
�sequential_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_hidden_layer_tensorarrayunstack_tensorlistfromtensor�sequential_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2|
<sequential/Hidden_layer/while/gru_cell/MatMul/ReadVariableOp<sequential/Hidden_layer/while/gru_cell/MatMul/ReadVariableOp2�
>sequential/Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp>sequential/Hidden_layer/while/gru_cell/MatMul_1/ReadVariableOp2n
5sequential/Hidden_layer/while/gru_cell/ReadVariableOp5sequential/Hidden_layer/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_10084028
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10084028___redundant_placeholder06
2while_while_cond_10084028___redundant_placeholder16
2while_while_cond_10084028___redundant_placeholder26
2while_while_cond_10084028___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
input_layer8
serving_default_input_layer:0���������d@
Output_layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�d
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_sequential
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
iter

beta_1

beta_2
	 decay
!learning_ratemLmM"mN#mO$mPvQvR"vS#vT$vU"
	optimizer
C
"0
#1
$2
3
4"
trackable_list_wrapper
C
"0
#1
$2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_sequential_layer_call_fn_10082593
-__inference_sequential_layer_call_fn_10082948
-__inference_sequential_layer_call_fn_10082963
-__inference_sequential_layer_call_fn_10082895�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_sequential_layer_call_and_return_conditional_losses_10083130
H__inference_sequential_layer_call_and_return_conditional_losses_10083345
H__inference_sequential_layer_call_and_return_conditional_losses_10082911
H__inference_sequential_layer_call_and_return_conditional_losses_10082927�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
#__inference__wrapped_model_10082018input_layer"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
*serving_default"
signature_map
�

"kernel
#recurrent_kernel
$bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

2states
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
/__inference_Hidden_layer_layer_call_fn_10083373
/__inference_Hidden_layer_layer_call_fn_10083384
/__inference_Hidden_layer_layer_call_fn_10083395
/__inference_Hidden_layer_layer_call_fn_10083406�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10083567
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10083776
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10083937
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10084146�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
%:#d2Output_layer/kernel
:2Output_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_Output_layer_layer_call_fn_10084155�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_Output_layer_layer_call_and_return_conditional_losses_10084165�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-	�2Hidden_layer/gru_cell/kernel
9:7	d�2&Hidden_layer/gru_cell/recurrent_kernel
-:+	�2Hidden_layer/gru_cell/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_signature_wrapper_10083362input_layer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
"0
#1
$2"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
+__inference_gru_cell_layer_call_fn_10084179
+__inference_gru_cell_layer_call_fn_10084193�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_gru_cell_layer_call_and_return_conditional_losses_10084236
F__inference_gru_cell_layer_call_and_return_conditional_losses_10084303�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Dtotal
	Ecount
F	variables
G	keras_api"
_tf_keras_metric
N
	Htotal
	Icount
J	variables
K	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
D0
E1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
:  (2total
:  (2count
.
H0
I1"
trackable_list_wrapper
-
J	variables"
_generic_user_object
*:(d2Adam/Output_layer/kernel/m
$:"2Adam/Output_layer/bias/m
4:2	�2#Adam/Hidden_layer/gru_cell/kernel/m
>:<	d�2-Adam/Hidden_layer/gru_cell/recurrent_kernel/m
2:0	�2!Adam/Hidden_layer/gru_cell/bias/m
*:(d2Adam/Output_layer/kernel/v
$:"2Adam/Output_layer/bias/v
4:2	�2#Adam/Hidden_layer/gru_cell/kernel/v
>:<	d�2-Adam/Hidden_layer/gru_cell/recurrent_kernel/v
2:0	�2!Adam/Hidden_layer/gru_cell/bias/v�
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10083567}$"#O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0���������d
� �
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10083776}$"#O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0���������d
� �
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10083937m$"#?�<
5�2
$�!
inputs���������d

 
p 

 
� "%�"
�
0���������d
� �
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_10084146m$"#?�<
5�2
$�!
inputs���������d

 
p

 
� "%�"
�
0���������d
� �
/__inference_Hidden_layer_layer_call_fn_10083373p$"#O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "����������d�
/__inference_Hidden_layer_layer_call_fn_10083384p$"#O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "����������d�
/__inference_Hidden_layer_layer_call_fn_10083395`$"#?�<
5�2
$�!
inputs���������d

 
p 

 
� "����������d�
/__inference_Hidden_layer_layer_call_fn_10083406`$"#?�<
5�2
$�!
inputs���������d

 
p

 
� "����������d�
J__inference_Output_layer_layer_call_and_return_conditional_losses_10084165\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� �
/__inference_Output_layer_layer_call_fn_10084155O/�,
%�"
 �
inputs���������d
� "�����������
#__inference__wrapped_model_10082018~$"#8�5
.�+
)�&
input_layer���������d
� ";�8
6
Output_layer&�#
Output_layer����������
F__inference_gru_cell_layer_call_and_return_conditional_losses_10084236�$"#\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������d
p 
� "R�O
H�E
�
0/0���������d
$�!
�
0/1/0���������d
� �
F__inference_gru_cell_layer_call_and_return_conditional_losses_10084303�$"#\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������d
p
� "R�O
H�E
�
0/0���������d
$�!
�
0/1/0���������d
� �
+__inference_gru_cell_layer_call_fn_10084179�$"#\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������d
p 
� "D�A
�
0���������d
"�
�
1/0���������d�
+__inference_gru_cell_layer_call_fn_10084193�$"#\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������d
p
� "D�A
�
0���������d
"�
�
1/0���������d�
H__inference_sequential_layer_call_and_return_conditional_losses_10082911p$"#@�=
6�3
)�&
input_layer���������d
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_layer_call_and_return_conditional_losses_10082927p$"#@�=
6�3
)�&
input_layer���������d
p

 
� "%�"
�
0���������
� �
H__inference_sequential_layer_call_and_return_conditional_losses_10083130k$"#;�8
1�.
$�!
inputs���������d
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_layer_call_and_return_conditional_losses_10083345k$"#;�8
1�.
$�!
inputs���������d
p

 
� "%�"
�
0���������
� �
-__inference_sequential_layer_call_fn_10082593c$"#@�=
6�3
)�&
input_layer���������d
p 

 
� "�����������
-__inference_sequential_layer_call_fn_10082895c$"#@�=
6�3
)�&
input_layer���������d
p

 
� "�����������
-__inference_sequential_layer_call_fn_10082948^$"#;�8
1�.
$�!
inputs���������d
p 

 
� "�����������
-__inference_sequential_layer_call_fn_10082963^$"#;�8
1�.
$�!
inputs���������d
p

 
� "�����������
&__inference_signature_wrapper_10083362�$"#G�D
� 
=�:
8
input_layer)�&
input_layer���������d";�8
6
Output_layer&�#
Output_layer���������