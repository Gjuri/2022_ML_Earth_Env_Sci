тн
∙╩
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Л
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
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
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
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
И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68╬к
В
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
Щ
Hidden_layer/gru_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*/
shared_name Hidden_layer/gru_cell_4/kernel
Т
2Hidden_layer/gru_cell_4/kernel/Read/ReadVariableOpReadVariableOpHidden_layer/gru_cell_4/kernel*
_output_shapes
:	м*
dtype0
н
(Hidden_layer/gru_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dм*9
shared_name*(Hidden_layer/gru_cell_4/recurrent_kernel
ж
<Hidden_layer/gru_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp(Hidden_layer/gru_cell_4/recurrent_kernel*
_output_shapes
:	dм*
dtype0
Х
Hidden_layer/gru_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*-
shared_nameHidden_layer/gru_cell_4/bias
О
0Hidden_layer/gru_cell_4/bias/Read/ReadVariableOpReadVariableOpHidden_layer/gru_cell_4/bias*
_output_shapes
:	м*
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
Р
Adam/Output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/Output_layer/kernel/m
Й
.Adam/Output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output_layer/kernel/m*
_output_shapes

:d*
dtype0
И
Adam/Output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_layer/bias/m
Б
,Adam/Output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output_layer/bias/m*
_output_shapes
:*
dtype0
з
%Adam/Hidden_layer/gru_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*6
shared_name'%Adam/Hidden_layer/gru_cell_4/kernel/m
а
9Adam/Hidden_layer/gru_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/Hidden_layer/gru_cell_4/kernel/m*
_output_shapes
:	м*
dtype0
╗
/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dм*@
shared_name1/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/m
┤
CAdam/Hidden_layer/gru_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/m*
_output_shapes
:	dм*
dtype0
г
#Adam/Hidden_layer/gru_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*4
shared_name%#Adam/Hidden_layer/gru_cell_4/bias/m
Ь
7Adam/Hidden_layer/gru_cell_4/bias/m/Read/ReadVariableOpReadVariableOp#Adam/Hidden_layer/gru_cell_4/bias/m*
_output_shapes
:	м*
dtype0
Р
Adam/Output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/Output_layer/kernel/v
Й
.Adam/Output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output_layer/kernel/v*
_output_shapes

:d*
dtype0
И
Adam/Output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_layer/bias/v
Б
,Adam/Output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output_layer/bias/v*
_output_shapes
:*
dtype0
з
%Adam/Hidden_layer/gru_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*6
shared_name'%Adam/Hidden_layer/gru_cell_4/kernel/v
а
9Adam/Hidden_layer/gru_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/Hidden_layer/gru_cell_4/kernel/v*
_output_shapes
:	м*
dtype0
╗
/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dм*@
shared_name1/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/v
┤
CAdam/Hidden_layer/gru_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/v*
_output_shapes
:	dм*
dtype0
г
#Adam/Hidden_layer/gru_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*4
shared_name%#Adam/Hidden_layer/gru_cell_4/bias/v
Ь
7Adam/Hidden_layer/gru_cell_4/bias/v/Read/ReadVariableOpReadVariableOp#Adam/Hidden_layer/gru_cell_4/bias/v*
_output_shapes
:	м*
dtype0

NoOpNoOp
П+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╩*
value└*B╜* B╢*
Ъ
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
┴
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
ж

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Ю
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
░
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
╙

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
Я

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
У
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
^X
VARIABLE_VALUEHidden_layer/gru_cell_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(Hidden_layer/gru_cell_4/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEHidden_layer/gru_cell_4/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
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
У
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
ЗА
VARIABLE_VALUEAdam/Output_layer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/Output_layer/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE%Adam/Hidden_layer/gru_cell_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/Hidden_layer/gru_cell_4/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUEAdam/Output_layer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/Output_layer/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE%Adam/Hidden_layer/gru_cell_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/Hidden_layer/gru_cell_4/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ж
serving_default_input_layerPlaceholder*+
_output_shapes
:         d*
dtype0* 
shape:         d
╥
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerHidden_layer/gru_cell_4/biasHidden_layer/gru_cell_4/kernel(Hidden_layer/gru_cell_4/recurrent_kernelOutput_layer/kernelOutput_layer/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_11123653
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ё

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'Output_layer/kernel/Read/ReadVariableOp%Output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp2Hidden_layer/gru_cell_4/kernel/Read/ReadVariableOp<Hidden_layer/gru_cell_4/recurrent_kernel/Read/ReadVariableOp0Hidden_layer/gru_cell_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/Output_layer/kernel/m/Read/ReadVariableOp,Adam/Output_layer/bias/m/Read/ReadVariableOp9Adam/Hidden_layer/gru_cell_4/kernel/m/Read/ReadVariableOpCAdam/Hidden_layer/gru_cell_4/recurrent_kernel/m/Read/ReadVariableOp7Adam/Hidden_layer/gru_cell_4/bias/m/Read/ReadVariableOp.Adam/Output_layer/kernel/v/Read/ReadVariableOp,Adam/Output_layer/bias/v/Read/ReadVariableOp9Adam/Hidden_layer/gru_cell_4/kernel/v/Read/ReadVariableOpCAdam/Hidden_layer/gru_cell_4/recurrent_kernel/v/Read/ReadVariableOp7Adam/Hidden_layer/gru_cell_4/bias/v/Read/ReadVariableOpConst*%
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
GPU 2J 8В **
f%R#
!__inference__traced_save_11124689
Л
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameOutput_layer/kernelOutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateHidden_layer/gru_cell_4/kernel(Hidden_layer/gru_cell_4/recurrent_kernelHidden_layer/gru_cell_4/biastotalcounttotal_1count_1Adam/Output_layer/kernel/mAdam/Output_layer/bias/m%Adam/Hidden_layer/gru_cell_4/kernel/m/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/m#Adam/Hidden_layer/gru_cell_4/bias/mAdam/Output_layer/kernel/vAdam/Output_layer/bias/v%Adam/Hidden_layer/gru_cell_4/kernel/v/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/v#Adam/Hidden_layer/gru_cell_4/bias/v*$
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
GPU 2J 8В *-
f(R&
$__inference__traced_restore_11124771╩╡
б
┬
/__inference_Hidden_layer_layer_call_fn_11123675
inputs_0
unknown:	м
	unknown_0:	м
	unknown_1:	dм
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11122670o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
═	
√
J__inference_Output_layer_layer_call_and_return_conditional_losses_11124456

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ю4
О
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11122460

inputs&
gru_cell_4_11122384:	м&
gru_cell_4_11122386:	м&
gru_cell_4_11122388:	dм
identityИв"gru_cell_4/StatefulPartitionedCallвwhile;
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
valueB:╤
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
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask╠
"gru_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_4_11122384gru_cell_4_11122386gru_cell_4_11122388*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11122383n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_4_11122384gru_cell_4_11122386gru_cell_4_11122388*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11122396*
condR
while_cond_11122395*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         ds
NoOpNoOp#^gru_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"gru_cell_4/StatefulPartitionedCall"gru_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
И 
▄
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11124527

inputs
states_0*
readvariableop_resource:	м1
matmul_readvariableop_resource:	м3
 matmul_1_readvariableop_resource:	dм
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:         g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	м*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:         u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d_
mul_1MulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         d[
add_2AddV2split:output:2	mul_1:z:0*
T0*'
_output_shapes
:         dI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:         dU
mul_2MulSigmoid:y:0states_0*
T0*'
_output_shapes
:         dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         d[
mul_3Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:         dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         dЙ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         d
"
_user_specified_name
states/0
▌
Ў
/__inference_sequential_4_layer_call_fn_11123254

inputs
unknown:	м
	unknown_0:	м
	unknown_1:	dм
	unknown_2:d
	unknown_3:
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
Є]
Б
while_body_11123950
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	мD
1while_gru_cell_4_matmul_readvariableop_resource_0:	мF
3while_gru_cell_4_matmul_1_readvariableop_resource_0:	dм
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	мB
/while_gru_cell_4_matmul_readvariableop_resource:	мD
1while_gru_cell_4_matmul_1_readvariableop_resource:	dмИв&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0А
 while/gru_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:e
 while/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
while/gru_cell_4/ones_likeFill)while/gru_cell_4/ones_like/Shape:output:0)while/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         c
while/gru_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?г
while/gru_cell_4/dropout/MulMul#while/gru_cell_4/ones_like:output:0'while/gru_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:         q
while/gru_cell_4/dropout/ShapeShape#while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:║
5while/gru_cell_4/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*l
'while/gru_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>┘
%while/gru_cell_4/dropout/GreaterEqualGreaterEqual>while/gru_cell_4/dropout/random_uniform/RandomUniform:output:00while/gru_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         С
while/gru_cell_4/dropout/CastCast)while/gru_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         Ь
while/gru_cell_4/dropout/Mul_1Mul while/gru_cell_4/dropout/Mul:z:0!while/gru_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:         e
 while/gru_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?з
while/gru_cell_4/dropout_1/MulMul#while/gru_cell_4/ones_like:output:0)while/gru_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:         s
 while/gru_cell_4/dropout_1/ShapeShape#while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:╦
7while/gru_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2n
)while/gru_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>▀
'while/gru_cell_4/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_4/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Х
while/gru_cell_4/dropout_1/CastCast+while/gru_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         в
 while/gru_cell_4/dropout_1/Mul_1Mul"while/gru_cell_4/dropout_1/Mul:z:0#while/gru_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         e
 while/gru_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?з
while/gru_cell_4/dropout_2/MulMul#while/gru_cell_4/ones_like:output:0)while/gru_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:         s
 while/gru_cell_4/dropout_2/ShapeShape#while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:╦
7while/gru_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2n
)while/gru_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>▀
'while/gru_cell_4/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_4/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Х
while/gru_cell_4/dropout_2/CastCast+while/gru_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         в
 while/gru_cell_4/dropout_2/Mul_1Mul"while/gru_cell_4/dropout_2/Mul:z:0#while/gru_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numг
while/gru_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Щ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Ю
while/gru_cell_4/MatMulMatMulwhile/gru_cell_4/mul:z:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мЬ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╓
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitЭ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	dм*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ма
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         К
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitУ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         do
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         dХ
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         ds
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dТ
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         dО
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dk
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dВ
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:         d[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         dО
while/gru_cell_4/mul_3Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dЙ
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_2:z:0while/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         d├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥w
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         d┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
╣k
М
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11124067
inputs_05
"gru_cell_4_readvariableop_resource:	м<
)gru_cell_4_matmul_readvariableop_resource:	м>
+gru_cell_4_matmul_1_readvariableop_resource:	dм
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhile=
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
valueB:╤
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
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskb
gru_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:_
gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
gru_cell_4/ones_likeFill#gru_cell_4/ones_like/Shape:output:0#gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         ]
gru_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?С
gru_cell_4/dropout/MulMulgru_cell_4/ones_like:output:0!gru_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:         e
gru_cell_4/dropout/ShapeShapegru_cell_4/ones_like:output:0*
T0*
_output_shapes
:о
/gru_cell_4/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*f
!gru_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>╟
gru_cell_4/dropout/GreaterEqualGreaterEqual8gru_cell_4/dropout/random_uniform/RandomUniform:output:0*gru_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
gru_cell_4/dropout/CastCast#gru_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         К
gru_cell_4/dropout/Mul_1Mulgru_cell_4/dropout/Mul:z:0gru_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:         _
gru_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?Х
gru_cell_4/dropout_1/MulMulgru_cell_4/ones_like:output:0#gru_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:         g
gru_cell_4/dropout_1/ShapeShapegru_cell_4/ones_like:output:0*
T0*
_output_shapes
:┐
1gru_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2h
#gru_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>═
!gru_cell_4/dropout_1/GreaterEqualGreaterEqual:gru_cell_4/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Й
gru_cell_4/dropout_1/CastCast%gru_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         Р
gru_cell_4/dropout_1/Mul_1Mulgru_cell_4/dropout_1/Mul:z:0gru_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         _
gru_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?Х
gru_cell_4/dropout_2/MulMulgru_cell_4/ones_like:output:0#gru_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:         g
gru_cell_4/dropout_2/ShapeShapegru_cell_4/ones_like:output:0*
T0*
_output_shapes
:┐
1gru_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2h
#gru_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>═
!gru_cell_4/dropout_2/GreaterEqualGreaterEqual:gru_cell_4/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Й
gru_cell_4/dropout_2/CastCast%gru_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         Р
gru_cell_4/dropout_2/Mul_1Mulgru_cell_4/dropout_2/Mul:z:0gru_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         }
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	м*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
num
gru_cell_4/mulMulstrided_slice_2:output:0gru_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Л
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype0М
gru_cell_4/MatMulMatMulgru_cell_4/mul:z:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мК
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ─
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitП
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мО
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Є
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitБ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dc
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:         dГ
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dg
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dА
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d|
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         d_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dq
gru_cell_4/mul_2Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:         dU
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d|
gru_cell_4/mul_3Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dw
gru_cell_4/add_3AddV2gru_cell_4/mul_2:z:0gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┐
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11123950*
condR
while_cond_11123949*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
Є]
Б
while_body_11124320
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	мD
1while_gru_cell_4_matmul_readvariableop_resource_0:	мF
3while_gru_cell_4_matmul_1_readvariableop_resource_0:	dм
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	мB
/while_gru_cell_4_matmul_readvariableop_resource:	мD
1while_gru_cell_4_matmul_1_readvariableop_resource:	dмИв&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0А
 while/gru_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:e
 while/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
while/gru_cell_4/ones_likeFill)while/gru_cell_4/ones_like/Shape:output:0)while/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         c
while/gru_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?г
while/gru_cell_4/dropout/MulMul#while/gru_cell_4/ones_like:output:0'while/gru_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:         q
while/gru_cell_4/dropout/ShapeShape#while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:║
5while/gru_cell_4/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*l
'while/gru_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>┘
%while/gru_cell_4/dropout/GreaterEqualGreaterEqual>while/gru_cell_4/dropout/random_uniform/RandomUniform:output:00while/gru_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         С
while/gru_cell_4/dropout/CastCast)while/gru_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         Ь
while/gru_cell_4/dropout/Mul_1Mul while/gru_cell_4/dropout/Mul:z:0!while/gru_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:         e
 while/gru_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?з
while/gru_cell_4/dropout_1/MulMul#while/gru_cell_4/ones_like:output:0)while/gru_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:         s
 while/gru_cell_4/dropout_1/ShapeShape#while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:╦
7while/gru_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2n
)while/gru_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>▀
'while/gru_cell_4/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_4/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Х
while/gru_cell_4/dropout_1/CastCast+while/gru_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         в
 while/gru_cell_4/dropout_1/Mul_1Mul"while/gru_cell_4/dropout_1/Mul:z:0#while/gru_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         e
 while/gru_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?з
while/gru_cell_4/dropout_2/MulMul#while/gru_cell_4/ones_like:output:0)while/gru_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:         s
 while/gru_cell_4/dropout_2/ShapeShape#while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:╦
7while/gru_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2n
)while/gru_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>▀
'while/gru_cell_4/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_4/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Х
while/gru_cell_4/dropout_2/CastCast+while/gru_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         в
 while/gru_cell_4/dropout_2/Mul_1Mul"while/gru_cell_4/dropout_2/Mul:z:0#while/gru_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numг
while/gru_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Щ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Ю
while/gru_cell_4/MatMulMatMulwhile/gru_cell_4/mul:z:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мЬ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╓
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitЭ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	dм*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ма
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         К
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitУ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         do
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         dХ
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         ds
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dТ
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         dО
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dk
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dВ
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:         d[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         dО
while/gru_cell_4/mul_3Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dЙ
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_2:z:0while/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         d├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥w
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         d┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
ь
√
/__inference_sequential_4_layer_call_fn_11122884
input_layer
unknown:	м
	unknown_0:	м
	unknown_1:	dм
	unknown_2:d
	unknown_3:
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_11122871o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:         d
%
_user_specified_nameinput_layer
ї@
Б
while_body_11122753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	мD
1while_gru_cell_4_matmul_readvariableop_resource_0:	мF
3while_gru_cell_4_matmul_1_readvariableop_resource_0:	dм
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	мB
/while_gru_cell_4_matmul_readvariableop_resource:	мD
1while_gru_cell_4_matmul_1_readvariableop_resource:	dмИв&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0А
 while/gru_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:e
 while/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
while/gru_cell_4/ones_likeFill)while/gru_cell_4/ones_like/Shape:output:0)while/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numд
while/gru_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         Щ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Ю
while/gru_cell_4/MatMulMatMulwhile/gru_cell_4/mul:z:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мЬ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╓
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitЭ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	dм*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ма
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         К
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitУ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         do
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         dХ
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         ds
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dТ
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         dО
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dk
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dВ
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:         d[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         dО
while/gru_cell_4/mul_3Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dЙ
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_2:z:0while/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         d├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥w
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         d┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
Є]
Б
while_body_11122999
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	мD
1while_gru_cell_4_matmul_readvariableop_resource_0:	мF
3while_gru_cell_4_matmul_1_readvariableop_resource_0:	dм
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	мB
/while_gru_cell_4_matmul_readvariableop_resource:	мD
1while_gru_cell_4_matmul_1_readvariableop_resource:	dмИв&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0А
 while/gru_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:e
 while/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
while/gru_cell_4/ones_likeFill)while/gru_cell_4/ones_like/Shape:output:0)while/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         c
while/gru_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?г
while/gru_cell_4/dropout/MulMul#while/gru_cell_4/ones_like:output:0'while/gru_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:         q
while/gru_cell_4/dropout/ShapeShape#while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:║
5while/gru_cell_4/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*l
'while/gru_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>┘
%while/gru_cell_4/dropout/GreaterEqualGreaterEqual>while/gru_cell_4/dropout/random_uniform/RandomUniform:output:00while/gru_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         С
while/gru_cell_4/dropout/CastCast)while/gru_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         Ь
while/gru_cell_4/dropout/Mul_1Mul while/gru_cell_4/dropout/Mul:z:0!while/gru_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:         e
 while/gru_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?з
while/gru_cell_4/dropout_1/MulMul#while/gru_cell_4/ones_like:output:0)while/gru_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:         s
 while/gru_cell_4/dropout_1/ShapeShape#while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:╦
7while/gru_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2n
)while/gru_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>▀
'while/gru_cell_4/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_4/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Х
while/gru_cell_4/dropout_1/CastCast+while/gru_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         в
 while/gru_cell_4/dropout_1/Mul_1Mul"while/gru_cell_4/dropout_1/Mul:z:0#while/gru_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         e
 while/gru_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?з
while/gru_cell_4/dropout_2/MulMul#while/gru_cell_4/ones_like:output:0)while/gru_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:         s
 while/gru_cell_4/dropout_2/ShapeShape#while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:╦
7while/gru_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2n
)while/gru_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>▀
'while/gru_cell_4/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_4/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Х
while/gru_cell_4/dropout_2/CastCast+while/gru_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         в
 while/gru_cell_4/dropout_2/Mul_1Mul"while/gru_cell_4/dropout_2/Mul:z:0#while/gru_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numг
while/gru_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Щ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Ю
while/gru_cell_4/MatMulMatMulwhile/gru_cell_4/mul:z:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мЬ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╓
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitЭ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	dм*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ма
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         К
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitУ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         do
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         dХ
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         ds
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dТ
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         dО
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dk
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dВ
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:         d[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         dО
while/gru_cell_4/mul_3Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dЙ
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_2:z:0while/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         d├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥w
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         d┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
ф
┤
while_cond_11123949
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11123949___redundant_placeholder06
2while_while_cond_11123949___redundant_placeholder16
2while_while_cond_11123949___redundant_placeholder26
2while_while_cond_11123949___redundant_placeholder3
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
 b
у
$__inference__traced_restore_11124771
file_prefix6
$assignvariableop_output_layer_kernel:d2
$assignvariableop_1_output_layer_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: D
1assignvariableop_7_hidden_layer_gru_cell_4_kernel:	мN
;assignvariableop_8_hidden_layer_gru_cell_4_recurrent_kernel:	dмB
/assignvariableop_9_hidden_layer_gru_cell_4_bias:	м#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: @
.assignvariableop_14_adam_output_layer_kernel_m:d:
,assignvariableop_15_adam_output_layer_bias_m:L
9assignvariableop_16_adam_hidden_layer_gru_cell_4_kernel_m:	мV
Cassignvariableop_17_adam_hidden_layer_gru_cell_4_recurrent_kernel_m:	dмJ
7assignvariableop_18_adam_hidden_layer_gru_cell_4_bias_m:	м@
.assignvariableop_19_adam_output_layer_kernel_v:d:
,assignvariableop_20_adam_output_layer_bias_v:L
9assignvariableop_21_adam_hidden_layer_gru_cell_4_kernel_v:	мV
Cassignvariableop_22_adam_hidden_layer_gru_cell_4_recurrent_kernel_v:	dмJ
7assignvariableop_23_adam_hidden_layer_gru_cell_4_bias_v:	м
identity_25ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Ф
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*║
value░BнB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHв
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOpAssignVariableOp$assignvariableop_output_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_1AssignVariableOp$assignvariableop_1_output_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_7AssignVariableOp1assignvariableop_7_hidden_layer_gru_cell_4_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_8AssignVariableOp;assignvariableop_8_hidden_layer_gru_cell_4_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_9AssignVariableOp/assignvariableop_9_hidden_layer_gru_cell_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_14AssignVariableOp.assignvariableop_14_adam_output_layer_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_output_layer_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_16AssignVariableOp9assignvariableop_16_adam_hidden_layer_gru_cell_4_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_17AssignVariableOpCassignvariableop_17_adam_hidden_layer_gru_cell_4_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_18AssignVariableOp7assignvariableop_18_adam_hidden_layer_gru_cell_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_output_layer_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_output_layer_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_21AssignVariableOp9assignvariableop_21_adam_hidden_layer_gru_cell_4_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_22AssignVariableOpCassignvariableop_22_adam_hidden_layer_gru_cell_4_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_23AssignVariableOp7assignvariableop_23_adam_hidden_layer_gru_cell_4_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ▀
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ╠
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
▌
Г
J__inference_sequential_4_layer_call_and_return_conditional_losses_11122871

inputs(
hidden_layer_11122847:	м(
hidden_layer_11122849:	м(
hidden_layer_11122851:	dм'
output_layer_11122865:d#
output_layer_11122867:
identityИв$Hidden_layer/StatefulPartitionedCallв$Output_layer/StatefulPartitionedCallЯ
$Hidden_layer/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_11122847hidden_layer_11122849hidden_layer_11122851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11122846н
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall-Hidden_layer/StatefulPartitionedCall:output:0output_layer_11122865output_layer_11122867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_11122864|
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ф
NoOpNoOp%^Hidden_layer/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 2L
$Hidden_layer/StatefulPartitionedCall$Hidden_layer/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
─

л
 Hidden_layer_while_cond_111235126
2hidden_layer_while_hidden_layer_while_loop_counter<
8hidden_layer_while_hidden_layer_while_maximum_iterations"
hidden_layer_while_placeholder$
 hidden_layer_while_placeholder_1$
 hidden_layer_while_placeholder_28
4hidden_layer_while_less_hidden_layer_strided_slice_1P
Lhidden_layer_while_hidden_layer_while_cond_11123512___redundant_placeholder0P
Lhidden_layer_while_hidden_layer_while_cond_11123512___redundant_placeholder1P
Lhidden_layer_while_hidden_layer_while_cond_11123512___redundant_placeholder2P
Lhidden_layer_while_hidden_layer_while_cond_11123512___redundant_placeholder3
hidden_layer_while_identity
Ц
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
ф
┤
while_cond_11122395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11122395___redundant_placeholder06
2while_while_cond_11122395___redundant_placeholder16
2while_while_cond_11122395___redundant_placeholder26
2while_while_cond_11122395___redundant_placeholder3
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
╬
Ь
/__inference_Output_layer_layer_call_fn_11124446

inputs
unknown:d
	unknown_0:
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_11122864o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ю4
О
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11122670

inputs&
gru_cell_4_11122594:	м&
gru_cell_4_11122596:	м&
gru_cell_4_11122598:	dм
identityИв"gru_cell_4/StatefulPartitionedCallвwhile;
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
valueB:╤
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
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask╠
"gru_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_4_11122594gru_cell_4_11122596gru_cell_4_11122598*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11122554n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_4_11122594gru_cell_4_11122596gru_cell_4_11122598*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11122606*
condR
while_cond_11122605*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         ds
NoOpNoOp#^gru_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"gru_cell_4/StatefulPartitionedCall"gru_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ш 
╝
while_body_11122606
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_4_11122628_0:	м.
while_gru_cell_4_11122630_0:	м.
while_gru_cell_4_11122632_0:	dм
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_4_11122628:	м,
while_gru_cell_4_11122630:	м,
while_gru_cell_4_11122632:	dмИв(while/gru_cell_4/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0З
(while/gru_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_4_11122628_0while_gru_cell_4_11122630_0while_gru_cell_4_11122632_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11122554┌
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥О
while/Identity_4Identity1while/gru_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         dw

while/NoOpNoOp)^while/gru_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_4_11122628while_gru_cell_4_11122628_0"8
while_gru_cell_4_11122630while_gru_cell_4_11122630_0"8
while_gru_cell_4_11122632while_gru_cell_4_11122632_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2T
(while/gru_cell_4/StatefulPartitionedCall(while/gru_cell_4/StatefulPartitionedCall: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
ФR
З
 Hidden_layer_while_body_111233226
2hidden_layer_while_hidden_layer_while_loop_counter<
8hidden_layer_while_hidden_layer_while_maximum_iterations"
hidden_layer_while_placeholder$
 hidden_layer_while_placeholder_1$
 hidden_layer_while_placeholder_25
1hidden_layer_while_hidden_layer_strided_slice_1_0q
mhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0J
7hidden_layer_while_gru_cell_4_readvariableop_resource_0:	мQ
>hidden_layer_while_gru_cell_4_matmul_readvariableop_resource_0:	мS
@hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource_0:	dм
hidden_layer_while_identity!
hidden_layer_while_identity_1!
hidden_layer_while_identity_2!
hidden_layer_while_identity_3!
hidden_layer_while_identity_43
/hidden_layer_while_hidden_layer_strided_slice_1o
khidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensorH
5hidden_layer_while_gru_cell_4_readvariableop_resource:	мO
<hidden_layer_while_gru_cell_4_matmul_readvariableop_resource:	мQ
>hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource:	dмИв3Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOpв5Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOpв,Hidden_layer/while/gru_cell_4/ReadVariableOpХ
DHidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ч
6Hidden_layer/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0hidden_layer_while_placeholderMHidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ъ
-Hidden_layer/while/gru_cell_4/ones_like/ShapeShape=Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:r
-Hidden_layer/while/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╤
'Hidden_layer/while/gru_cell_4/ones_likeFill6Hidden_layer/while/gru_cell_4/ones_like/Shape:output:06Hidden_layer/while/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         е
,Hidden_layer/while/gru_cell_4/ReadVariableOpReadVariableOp7hidden_layer_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Э
%Hidden_layer/while/gru_cell_4/unstackUnpack4Hidden_layer/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
num╦
!Hidden_layer/while/gru_cell_4/mulMul=Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:00Hidden_layer/while/gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         │
3Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp>hidden_layer_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	м*
dtype0┼
$Hidden_layer/while/gru_cell_4/MatMulMatMul%Hidden_layer/while/gru_cell_4/mul:z:0;Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м├
%Hidden_layer/while/gru_cell_4/BiasAddBiasAdd.Hidden_layer/while/gru_cell_4/MatMul:product:0.Hidden_layer/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мx
-Hidden_layer/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ¤
#Hidden_layer/while/gru_cell_4/splitSplit6Hidden_layer/while/gru_cell_4/split/split_dim:output:0.Hidden_layer/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_split╖
5Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp@hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	dм*
dtype0─
&Hidden_layer/while/gru_cell_4/MatMul_1MatMul hidden_layer_while_placeholder_2=Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м╟
'Hidden_layer/while/gru_cell_4/BiasAdd_1BiasAdd0Hidden_layer/while/gru_cell_4/MatMul_1:product:0.Hidden_layer/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мx
#Hidden_layer/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       z
/Hidden_layer/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
%Hidden_layer/while/gru_cell_4/split_1SplitV0Hidden_layer/while/gru_cell_4/BiasAdd_1:output:0,Hidden_layer/while/gru_cell_4/Const:output:08Hidden_layer/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_split║
!Hidden_layer/while/gru_cell_4/addAddV2,Hidden_layer/while/gru_cell_4/split:output:0.Hidden_layer/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dЙ
%Hidden_layer/while/gru_cell_4/SigmoidSigmoid%Hidden_layer/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         d╝
#Hidden_layer/while/gru_cell_4/add_1AddV2,Hidden_layer/while/gru_cell_4/split:output:1.Hidden_layer/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dН
'Hidden_layer/while/gru_cell_4/Sigmoid_1Sigmoid'Hidden_layer/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         d╣
#Hidden_layer/while/gru_cell_4/mul_1Mul+Hidden_layer/while/gru_cell_4/Sigmoid_1:y:0.Hidden_layer/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d╡
#Hidden_layer/while/gru_cell_4/add_2AddV2,Hidden_layer/while/gru_cell_4/split:output:2'Hidden_layer/while/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dЕ
"Hidden_layer/while/gru_cell_4/ReluRelu'Hidden_layer/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dй
#Hidden_layer/while/gru_cell_4/mul_2Mul)Hidden_layer/while/gru_cell_4/Sigmoid:y:0 hidden_layer_while_placeholder_2*
T0*'
_output_shapes
:         dh
#Hidden_layer/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?│
!Hidden_layer/while/gru_cell_4/subSub,Hidden_layer/while/gru_cell_4/sub/x:output:0)Hidden_layer/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d╡
#Hidden_layer/while/gru_cell_4/mul_3Mul%Hidden_layer/while/gru_cell_4/sub:z:00Hidden_layer/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         d░
#Hidden_layer/while/gru_cell_4/add_3AddV2'Hidden_layer/while/gru_cell_4/mul_2:z:0'Hidden_layer/while/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dў
7Hidden_layer/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem hidden_layer_while_placeholder_1hidden_layer_while_placeholder'Hidden_layer/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥Z
Hidden_layer/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Г
Hidden_layer/while/addAddV2hidden_layer_while_placeholder!Hidden_layer/while/add/y:output:0*
T0*
_output_shapes
: \
Hidden_layer/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ы
Hidden_layer/while/add_1AddV22hidden_layer_while_hidden_layer_while_loop_counter#Hidden_layer/while/add_1/y:output:0*
T0*
_output_shapes
: А
Hidden_layer/while/IdentityIdentityHidden_layer/while/add_1:z:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: Ю
Hidden_layer/while/Identity_1Identity8hidden_layer_while_hidden_layer_while_maximum_iterations^Hidden_layer/while/NoOp*
T0*
_output_shapes
: А
Hidden_layer/while/Identity_2IdentityHidden_layer/while/add:z:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: └
Hidden_layer/while/Identity_3IdentityGHidden_layer/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: :щш╥Ю
Hidden_layer/while/Identity_4Identity'Hidden_layer/while/gru_cell_4/add_3:z:0^Hidden_layer/while/NoOp*
T0*'
_output_shapes
:         dЎ
Hidden_layer/while/NoOpNoOp4^Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp6^Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp-^Hidden_layer/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "В
>hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource@hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource_0"~
<hidden_layer_while_gru_cell_4_matmul_readvariableop_resource>hidden_layer_while_gru_cell_4_matmul_readvariableop_resource_0"p
5hidden_layer_while_gru_cell_4_readvariableop_resource7hidden_layer_while_gru_cell_4_readvariableop_resource_0"d
/hidden_layer_while_hidden_layer_strided_slice_11hidden_layer_while_hidden_layer_strided_slice_1_0"C
hidden_layer_while_identity$Hidden_layer/while/Identity:output:0"G
hidden_layer_while_identity_1&Hidden_layer/while/Identity_1:output:0"G
hidden_layer_while_identity_2&Hidden_layer/while/Identity_2:output:0"G
hidden_layer_while_identity_3&Hidden_layer/while/Identity_3:output:0"G
hidden_layer_while_identity_4&Hidden_layer/while/Identity_4:output:0"▄
khidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensormhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2j
3Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp3Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp2n
5Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp5Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp2\
,Hidden_layer/while/gru_cell_4/ReadVariableOp,Hidden_layer/while/gru_cell_4/ReadVariableOp: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
ф
┤
while_cond_11124134
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11124134___redundant_placeholder06
2while_while_cond_11124134___redundant_placeholder16
2while_while_cond_11124134___redundant_placeholder26
2while_while_cond_11124134___redundant_placeholder3
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
▌
Г
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123158

inputs(
hidden_layer_11123145:	м(
hidden_layer_11123147:	м(
hidden_layer_11123149:	dм'
output_layer_11123152:d#
output_layer_11123154:
identityИв$Hidden_layer/StatefulPartitionedCallв$Output_layer/StatefulPartitionedCallЯ
$Hidden_layer/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_11123145hidden_layer_11123147hidden_layer_11123149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11123116н
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall-Hidden_layer/StatefulPartitionedCall:output:0output_layer_11123152output_layer_11123154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_11122864|
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ф
NoOpNoOp%^Hidden_layer/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 2L
$Hidden_layer/StatefulPartitionedCall$Hidden_layer/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
ф
┤
while_cond_11122752
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11122752___redundant_placeholder06
2while_while_cond_11122752___redundant_placeholder16
2while_while_cond_11122752___redundant_placeholder26
2while_while_cond_11122752___redundant_placeholder3
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
╢

▄
-__inference_gru_cell_4_layer_call_fn_11124470

inputs
states_0
unknown:	м
	unknown_0:	м
	unknown_1:	dм
identity

identity_1ИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11122383o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         d
"
_user_specified_name
states/0
ф
┤
while_cond_11124319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11124319___redundant_placeholder06
2while_while_cond_11124319___redundant_placeholder16
2while_while_cond_11124319___redundant_placeholder26
2while_while_cond_11124319___redundant_placeholder3
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
ь
√
/__inference_sequential_4_layer_call_fn_11123186
input_layer
unknown:	м
	unknown_0:	м
	unknown_1:	dм
	unknown_2:d
	unknown_3:
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:         d
%
_user_specified_nameinput_layer
Дk
К
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11124437

inputs5
"gru_cell_4_readvariableop_resource:	м<
)gru_cell_4_matmul_readvariableop_resource:	м>
+gru_cell_4_matmul_1_readvariableop_resource:	dм
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhile;
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
valueB:╤
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
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskb
gru_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:_
gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
gru_cell_4/ones_likeFill#gru_cell_4/ones_like/Shape:output:0#gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         ]
gru_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?С
gru_cell_4/dropout/MulMulgru_cell_4/ones_like:output:0!gru_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:         e
gru_cell_4/dropout/ShapeShapegru_cell_4/ones_like:output:0*
T0*
_output_shapes
:о
/gru_cell_4/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*f
!gru_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>╟
gru_cell_4/dropout/GreaterEqualGreaterEqual8gru_cell_4/dropout/random_uniform/RandomUniform:output:0*gru_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
gru_cell_4/dropout/CastCast#gru_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         К
gru_cell_4/dropout/Mul_1Mulgru_cell_4/dropout/Mul:z:0gru_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:         _
gru_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?Х
gru_cell_4/dropout_1/MulMulgru_cell_4/ones_like:output:0#gru_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:         g
gru_cell_4/dropout_1/ShapeShapegru_cell_4/ones_like:output:0*
T0*
_output_shapes
:┐
1gru_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2h
#gru_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>═
!gru_cell_4/dropout_1/GreaterEqualGreaterEqual:gru_cell_4/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Й
gru_cell_4/dropout_1/CastCast%gru_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         Р
gru_cell_4/dropout_1/Mul_1Mulgru_cell_4/dropout_1/Mul:z:0gru_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         _
gru_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?Х
gru_cell_4/dropout_2/MulMulgru_cell_4/ones_like:output:0#gru_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:         g
gru_cell_4/dropout_2/ShapeShapegru_cell_4/ones_like:output:0*
T0*
_output_shapes
:┐
1gru_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2h
#gru_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>═
!gru_cell_4/dropout_2/GreaterEqualGreaterEqual:gru_cell_4/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Й
gru_cell_4/dropout_2/CastCast%gru_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         Р
gru_cell_4/dropout_2/Mul_1Mulgru_cell_4/dropout_2/Mul:z:0gru_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         }
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	м*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
num
gru_cell_4/mulMulstrided_slice_2:output:0gru_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Л
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype0М
gru_cell_4/MatMulMatMulgru_cell_4/mul:z:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мК
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ─
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitП
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мО
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Є
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitБ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dc
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:         dГ
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dg
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dА
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d|
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         d_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dq
gru_cell_4/mul_2Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:         dU
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d|
gru_cell_4/mul_3Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dw
gru_cell_4/add_3AddV2gru_cell_4/mul_2:z:0gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┐
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11124320*
condR
while_cond_11124319*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d         d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         dd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         d: : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
╕c
С
-sequential_4_Hidden_layer_while_body_11122210P
Lsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_loop_counterV
Rsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_maximum_iterations/
+sequential_4_hidden_layer_while_placeholder1
-sequential_4_hidden_layer_while_placeholder_11
-sequential_4_hidden_layer_while_placeholder_2O
Ksequential_4_hidden_layer_while_sequential_4_hidden_layer_strided_slice_1_0М
Зsequential_4_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_4_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0W
Dsequential_4_hidden_layer_while_gru_cell_4_readvariableop_resource_0:	м^
Ksequential_4_hidden_layer_while_gru_cell_4_matmul_readvariableop_resource_0:	м`
Msequential_4_hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource_0:	dм,
(sequential_4_hidden_layer_while_identity.
*sequential_4_hidden_layer_while_identity_1.
*sequential_4_hidden_layer_while_identity_2.
*sequential_4_hidden_layer_while_identity_3.
*sequential_4_hidden_layer_while_identity_4M
Isequential_4_hidden_layer_while_sequential_4_hidden_layer_strided_slice_1К
Еsequential_4_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_4_hidden_layer_tensorarrayunstack_tensorlistfromtensorU
Bsequential_4_hidden_layer_while_gru_cell_4_readvariableop_resource:	м\
Isequential_4_hidden_layer_while_gru_cell_4_matmul_readvariableop_resource:	м^
Ksequential_4_hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource:	dмИв@sequential_4/Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOpвBsequential_4/Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOpв9sequential_4/Hidden_layer/while/gru_cell_4/ReadVariableOpв
Qsequential_4/Hidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       й
Csequential_4/Hidden_layer/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЗsequential_4_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_4_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0+sequential_4_hidden_layer_while_placeholderZsequential_4/Hidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0┤
:sequential_4/Hidden_layer/while/gru_cell_4/ones_like/ShapeShapeJsequential_4/Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:
:sequential_4/Hidden_layer/while/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?°
4sequential_4/Hidden_layer/while/gru_cell_4/ones_likeFillCsequential_4/Hidden_layer/while/gru_cell_4/ones_like/Shape:output:0Csequential_4/Hidden_layer/while/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         ┐
9sequential_4/Hidden_layer/while/gru_cell_4/ReadVariableOpReadVariableOpDsequential_4_hidden_layer_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	м*
dtype0╖
2sequential_4/Hidden_layer/while/gru_cell_4/unstackUnpackAsequential_4/Hidden_layer/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numЄ
.sequential_4/Hidden_layer/while/gru_cell_4/mulMulJsequential_4/Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0=sequential_4/Hidden_layer/while/gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         ═
@sequential_4/Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOpKsequential_4_hidden_layer_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	м*
dtype0ь
1sequential_4/Hidden_layer/while/gru_cell_4/MatMulMatMul2sequential_4/Hidden_layer/while/gru_cell_4/mul:z:0Hsequential_4/Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мъ
2sequential_4/Hidden_layer/while/gru_cell_4/BiasAddBiasAdd;sequential_4/Hidden_layer/while/gru_cell_4/MatMul:product:0;sequential_4/Hidden_layer/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мЕ
:sequential_4/Hidden_layer/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         д
0sequential_4/Hidden_layer/while/gru_cell_4/splitSplitCsequential_4/Hidden_layer/while/gru_cell_4/split/split_dim:output:0;sequential_4/Hidden_layer/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_split╤
Bsequential_4/Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpMsequential_4_hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	dм*
dtype0ы
3sequential_4/Hidden_layer/while/gru_cell_4/MatMul_1MatMul-sequential_4_hidden_layer_while_placeholder_2Jsequential_4/Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мю
4sequential_4/Hidden_layer/while/gru_cell_4/BiasAdd_1BiasAdd=sequential_4/Hidden_layer/while/gru_cell_4/MatMul_1:product:0;sequential_4/Hidden_layer/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мЕ
0sequential_4/Hidden_layer/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       З
<sequential_4/Hidden_layer/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Є
2sequential_4/Hidden_layer/while/gru_cell_4/split_1SplitV=sequential_4/Hidden_layer/while/gru_cell_4/BiasAdd_1:output:09sequential_4/Hidden_layer/while/gru_cell_4/Const:output:0Esequential_4/Hidden_layer/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitс
.sequential_4/Hidden_layer/while/gru_cell_4/addAddV29sequential_4/Hidden_layer/while/gru_cell_4/split:output:0;sequential_4/Hidden_layer/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dг
2sequential_4/Hidden_layer/while/gru_cell_4/SigmoidSigmoid2sequential_4/Hidden_layer/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         dу
0sequential_4/Hidden_layer/while/gru_cell_4/add_1AddV29sequential_4/Hidden_layer/while/gru_cell_4/split:output:1;sequential_4/Hidden_layer/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dз
4sequential_4/Hidden_layer/while/gru_cell_4/Sigmoid_1Sigmoid4sequential_4/Hidden_layer/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dр
0sequential_4/Hidden_layer/while/gru_cell_4/mul_1Mul8sequential_4/Hidden_layer/while/gru_cell_4/Sigmoid_1:y:0;sequential_4/Hidden_layer/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d▄
0sequential_4/Hidden_layer/while/gru_cell_4/add_2AddV29sequential_4/Hidden_layer/while/gru_cell_4/split:output:24sequential_4/Hidden_layer/while/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dЯ
/sequential_4/Hidden_layer/while/gru_cell_4/ReluRelu4sequential_4/Hidden_layer/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         d╨
0sequential_4/Hidden_layer/while/gru_cell_4/mul_2Mul6sequential_4/Hidden_layer/while/gru_cell_4/Sigmoid:y:0-sequential_4_hidden_layer_while_placeholder_2*
T0*'
_output_shapes
:         du
0sequential_4/Hidden_layer/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┌
.sequential_4/Hidden_layer/while/gru_cell_4/subSub9sequential_4/Hidden_layer/while/gru_cell_4/sub/x:output:06sequential_4/Hidden_layer/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d▄
0sequential_4/Hidden_layer/while/gru_cell_4/mul_3Mul2sequential_4/Hidden_layer/while/gru_cell_4/sub:z:0=sequential_4/Hidden_layer/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         d╫
0sequential_4/Hidden_layer/while/gru_cell_4/add_3AddV24sequential_4/Hidden_layer/while/gru_cell_4/mul_2:z:04sequential_4/Hidden_layer/while/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dл
Dsequential_4/Hidden_layer/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_4_hidden_layer_while_placeholder_1+sequential_4_hidden_layer_while_placeholder4sequential_4/Hidden_layer/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥g
%sequential_4/Hidden_layer/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :к
#sequential_4/Hidden_layer/while/addAddV2+sequential_4_hidden_layer_while_placeholder.sequential_4/Hidden_layer/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_4/Hidden_layer/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╧
%sequential_4/Hidden_layer/while/add_1AddV2Lsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_loop_counter0sequential_4/Hidden_layer/while/add_1/y:output:0*
T0*
_output_shapes
: з
(sequential_4/Hidden_layer/while/IdentityIdentity)sequential_4/Hidden_layer/while/add_1:z:0%^sequential_4/Hidden_layer/while/NoOp*
T0*
_output_shapes
: ╥
*sequential_4/Hidden_layer/while/Identity_1IdentityRsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_maximum_iterations%^sequential_4/Hidden_layer/while/NoOp*
T0*
_output_shapes
: з
*sequential_4/Hidden_layer/while/Identity_2Identity'sequential_4/Hidden_layer/while/add:z:0%^sequential_4/Hidden_layer/while/NoOp*
T0*
_output_shapes
: ч
*sequential_4/Hidden_layer/while/Identity_3IdentityTsequential_4/Hidden_layer/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_4/Hidden_layer/while/NoOp*
T0*
_output_shapes
: :щш╥┼
*sequential_4/Hidden_layer/while/Identity_4Identity4sequential_4/Hidden_layer/while/gru_cell_4/add_3:z:0%^sequential_4/Hidden_layer/while/NoOp*
T0*'
_output_shapes
:         dк
$sequential_4/Hidden_layer/while/NoOpNoOpA^sequential_4/Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOpC^sequential_4/Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp:^sequential_4/Hidden_layer/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ь
Ksequential_4_hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resourceMsequential_4_hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource_0"Ш
Isequential_4_hidden_layer_while_gru_cell_4_matmul_readvariableop_resourceKsequential_4_hidden_layer_while_gru_cell_4_matmul_readvariableop_resource_0"К
Bsequential_4_hidden_layer_while_gru_cell_4_readvariableop_resourceDsequential_4_hidden_layer_while_gru_cell_4_readvariableop_resource_0"]
(sequential_4_hidden_layer_while_identity1sequential_4/Hidden_layer/while/Identity:output:0"a
*sequential_4_hidden_layer_while_identity_13sequential_4/Hidden_layer/while/Identity_1:output:0"a
*sequential_4_hidden_layer_while_identity_23sequential_4/Hidden_layer/while/Identity_2:output:0"a
*sequential_4_hidden_layer_while_identity_33sequential_4/Hidden_layer/while/Identity_3:output:0"a
*sequential_4_hidden_layer_while_identity_43sequential_4/Hidden_layer/while/Identity_4:output:0"Ш
Isequential_4_hidden_layer_while_sequential_4_hidden_layer_strided_slice_1Ksequential_4_hidden_layer_while_sequential_4_hidden_layer_strided_slice_1_0"Т
Еsequential_4_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_4_hidden_layer_tensorarrayunstack_tensorlistfromtensorЗsequential_4_hidden_layer_while_tensorarrayv2read_tensorlistgetitem_sequential_4_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2Д
@sequential_4/Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp@sequential_4/Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp2И
Bsequential_4/Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOpBsequential_4/Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp2v
9sequential_4/Hidden_layer/while/gru_cell_4/ReadVariableOp9sequential_4/Hidden_layer/while/gru_cell_4/ReadVariableOp: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
А 
┌
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11122383

inputs

states*
readvariableop_resource:	м1
matmul_readvariableop_resource:	м3
 matmul_1_readvariableop_resource:	dм
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:         g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	м*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:         u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d_
mul_1MulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         d[
add_2AddV2split:output:2	mul_1:z:0*
T0*'
_output_shapes
:         dI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_2MulSigmoid:y:0states*
T0*'
_output_shapes
:         dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         d[
mul_3Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:         dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         dЙ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         d
 
_user_specified_namestates
лt
З
 Hidden_layer_while_body_111235136
2hidden_layer_while_hidden_layer_while_loop_counter<
8hidden_layer_while_hidden_layer_while_maximum_iterations"
hidden_layer_while_placeholder$
 hidden_layer_while_placeholder_1$
 hidden_layer_while_placeholder_25
1hidden_layer_while_hidden_layer_strided_slice_1_0q
mhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0J
7hidden_layer_while_gru_cell_4_readvariableop_resource_0:	мQ
>hidden_layer_while_gru_cell_4_matmul_readvariableop_resource_0:	мS
@hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource_0:	dм
hidden_layer_while_identity!
hidden_layer_while_identity_1!
hidden_layer_while_identity_2!
hidden_layer_while_identity_3!
hidden_layer_while_identity_43
/hidden_layer_while_hidden_layer_strided_slice_1o
khidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensorH
5hidden_layer_while_gru_cell_4_readvariableop_resource:	мO
<hidden_layer_while_gru_cell_4_matmul_readvariableop_resource:	мQ
>hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource:	dмИв3Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOpв5Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOpв,Hidden_layer/while/gru_cell_4/ReadVariableOpХ
DHidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ч
6Hidden_layer/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0hidden_layer_while_placeholderMHidden_layer/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ъ
-Hidden_layer/while/gru_cell_4/ones_like/ShapeShape=Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:r
-Hidden_layer/while/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╤
'Hidden_layer/while/gru_cell_4/ones_likeFill6Hidden_layer/while/gru_cell_4/ones_like/Shape:output:06Hidden_layer/while/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         p
+Hidden_layer/while/gru_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?╩
)Hidden_layer/while/gru_cell_4/dropout/MulMul0Hidden_layer/while/gru_cell_4/ones_like:output:04Hidden_layer/while/gru_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:         Л
+Hidden_layer/while/gru_cell_4/dropout/ShapeShape0Hidden_layer/while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:╘
BHidden_layer/while/gru_cell_4/dropout/random_uniform/RandomUniformRandomUniform4Hidden_layer/while/gru_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*y
4Hidden_layer/while/gru_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>А
2Hidden_layer/while/gru_cell_4/dropout/GreaterEqualGreaterEqualKHidden_layer/while/gru_cell_4/dropout/random_uniform/RandomUniform:output:0=Hidden_layer/while/gru_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         л
*Hidden_layer/while/gru_cell_4/dropout/CastCast6Hidden_layer/while/gru_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ├
+Hidden_layer/while/gru_cell_4/dropout/Mul_1Mul-Hidden_layer/while/gru_cell_4/dropout/Mul:z:0.Hidden_layer/while/gru_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:         r
-Hidden_layer/while/gru_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?╬
+Hidden_layer/while/gru_cell_4/dropout_1/MulMul0Hidden_layer/while/gru_cell_4/ones_like:output:06Hidden_layer/while/gru_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:         Н
-Hidden_layer/while/gru_cell_4/dropout_1/ShapeShape0Hidden_layer/while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:х
DHidden_layer/while/gru_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform6Hidden_layer/while/gru_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2{
6Hidden_layer/while/gru_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>Ж
4Hidden_layer/while/gru_cell_4/dropout_1/GreaterEqualGreaterEqualMHidden_layer/while/gru_cell_4/dropout_1/random_uniform/RandomUniform:output:0?Hidden_layer/while/gru_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         п
,Hidden_layer/while/gru_cell_4/dropout_1/CastCast8Hidden_layer/while/gru_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ╔
-Hidden_layer/while/gru_cell_4/dropout_1/Mul_1Mul/Hidden_layer/while/gru_cell_4/dropout_1/Mul:z:00Hidden_layer/while/gru_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         r
-Hidden_layer/while/gru_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?╬
+Hidden_layer/while/gru_cell_4/dropout_2/MulMul0Hidden_layer/while/gru_cell_4/ones_like:output:06Hidden_layer/while/gru_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:         Н
-Hidden_layer/while/gru_cell_4/dropout_2/ShapeShape0Hidden_layer/while/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:х
DHidden_layer/while/gru_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform6Hidden_layer/while/gru_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2{
6Hidden_layer/while/gru_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>Ж
4Hidden_layer/while/gru_cell_4/dropout_2/GreaterEqualGreaterEqualMHidden_layer/while/gru_cell_4/dropout_2/random_uniform/RandomUniform:output:0?Hidden_layer/while/gru_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         п
,Hidden_layer/while/gru_cell_4/dropout_2/CastCast8Hidden_layer/while/gru_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ╔
-Hidden_layer/while/gru_cell_4/dropout_2/Mul_1Mul/Hidden_layer/while/gru_cell_4/dropout_2/Mul:z:00Hidden_layer/while/gru_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         е
,Hidden_layer/while/gru_cell_4/ReadVariableOpReadVariableOp7hidden_layer_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Э
%Hidden_layer/while/gru_cell_4/unstackUnpack4Hidden_layer/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
num╩
!Hidden_layer/while/gru_cell_4/mulMul=Hidden_layer/while/TensorArrayV2Read/TensorListGetItem:item:0/Hidden_layer/while/gru_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         │
3Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp>hidden_layer_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	м*
dtype0┼
$Hidden_layer/while/gru_cell_4/MatMulMatMul%Hidden_layer/while/gru_cell_4/mul:z:0;Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м├
%Hidden_layer/while/gru_cell_4/BiasAddBiasAdd.Hidden_layer/while/gru_cell_4/MatMul:product:0.Hidden_layer/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мx
-Hidden_layer/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ¤
#Hidden_layer/while/gru_cell_4/splitSplit6Hidden_layer/while/gru_cell_4/split/split_dim:output:0.Hidden_layer/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_split╖
5Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp@hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	dм*
dtype0─
&Hidden_layer/while/gru_cell_4/MatMul_1MatMul hidden_layer_while_placeholder_2=Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м╟
'Hidden_layer/while/gru_cell_4/BiasAdd_1BiasAdd0Hidden_layer/while/gru_cell_4/MatMul_1:product:0.Hidden_layer/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мx
#Hidden_layer/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       z
/Hidden_layer/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╛
%Hidden_layer/while/gru_cell_4/split_1SplitV0Hidden_layer/while/gru_cell_4/BiasAdd_1:output:0,Hidden_layer/while/gru_cell_4/Const:output:08Hidden_layer/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_split║
!Hidden_layer/while/gru_cell_4/addAddV2,Hidden_layer/while/gru_cell_4/split:output:0.Hidden_layer/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dЙ
%Hidden_layer/while/gru_cell_4/SigmoidSigmoid%Hidden_layer/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         d╝
#Hidden_layer/while/gru_cell_4/add_1AddV2,Hidden_layer/while/gru_cell_4/split:output:1.Hidden_layer/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dН
'Hidden_layer/while/gru_cell_4/Sigmoid_1Sigmoid'Hidden_layer/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         d╣
#Hidden_layer/while/gru_cell_4/mul_1Mul+Hidden_layer/while/gru_cell_4/Sigmoid_1:y:0.Hidden_layer/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d╡
#Hidden_layer/while/gru_cell_4/add_2AddV2,Hidden_layer/while/gru_cell_4/split:output:2'Hidden_layer/while/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dЕ
"Hidden_layer/while/gru_cell_4/ReluRelu'Hidden_layer/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dй
#Hidden_layer/while/gru_cell_4/mul_2Mul)Hidden_layer/while/gru_cell_4/Sigmoid:y:0 hidden_layer_while_placeholder_2*
T0*'
_output_shapes
:         dh
#Hidden_layer/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?│
!Hidden_layer/while/gru_cell_4/subSub,Hidden_layer/while/gru_cell_4/sub/x:output:0)Hidden_layer/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d╡
#Hidden_layer/while/gru_cell_4/mul_3Mul%Hidden_layer/while/gru_cell_4/sub:z:00Hidden_layer/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         d░
#Hidden_layer/while/gru_cell_4/add_3AddV2'Hidden_layer/while/gru_cell_4/mul_2:z:0'Hidden_layer/while/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dў
7Hidden_layer/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem hidden_layer_while_placeholder_1hidden_layer_while_placeholder'Hidden_layer/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥Z
Hidden_layer/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Г
Hidden_layer/while/addAddV2hidden_layer_while_placeholder!Hidden_layer/while/add/y:output:0*
T0*
_output_shapes
: \
Hidden_layer/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ы
Hidden_layer/while/add_1AddV22hidden_layer_while_hidden_layer_while_loop_counter#Hidden_layer/while/add_1/y:output:0*
T0*
_output_shapes
: А
Hidden_layer/while/IdentityIdentityHidden_layer/while/add_1:z:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: Ю
Hidden_layer/while/Identity_1Identity8hidden_layer_while_hidden_layer_while_maximum_iterations^Hidden_layer/while/NoOp*
T0*
_output_shapes
: А
Hidden_layer/while/Identity_2IdentityHidden_layer/while/add:z:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: └
Hidden_layer/while/Identity_3IdentityGHidden_layer/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^Hidden_layer/while/NoOp*
T0*
_output_shapes
: :щш╥Ю
Hidden_layer/while/Identity_4Identity'Hidden_layer/while/gru_cell_4/add_3:z:0^Hidden_layer/while/NoOp*
T0*'
_output_shapes
:         dЎ
Hidden_layer/while/NoOpNoOp4^Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp6^Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp-^Hidden_layer/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "В
>hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource@hidden_layer_while_gru_cell_4_matmul_1_readvariableop_resource_0"~
<hidden_layer_while_gru_cell_4_matmul_readvariableop_resource>hidden_layer_while_gru_cell_4_matmul_readvariableop_resource_0"p
5hidden_layer_while_gru_cell_4_readvariableop_resource7hidden_layer_while_gru_cell_4_readvariableop_resource_0"d
/hidden_layer_while_hidden_layer_strided_slice_11hidden_layer_while_hidden_layer_strided_slice_1_0"C
hidden_layer_while_identity$Hidden_layer/while/Identity:output:0"G
hidden_layer_while_identity_1&Hidden_layer/while/Identity_1:output:0"G
hidden_layer_while_identity_2&Hidden_layer/while/Identity_2:output:0"G
hidden_layer_while_identity_3&Hidden_layer/while/Identity_3:output:0"G
hidden_layer_while_identity_4&Hidden_layer/while/Identity_4:output:0"▄
khidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensormhidden_layer_while_tensorarrayv2read_tensorlistgetitem_hidden_layer_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2j
3Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp3Hidden_layer/while/gru_cell_4/MatMul/ReadVariableOp2n
5Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp5Hidden_layer/while/gru_cell_4/MatMul_1/ReadVariableOp2\
,Hidden_layer/while/gru_cell_4/ReadVariableOp,Hidden_layer/while/gru_cell_4/ReadVariableOp: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
▌
Ў
/__inference_sequential_4_layer_call_fn_11123239

inputs
unknown:	м
	unknown_0:	м
	unknown_1:	dм
	unknown_2:d
	unknown_3:
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_11122871o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
ф
┤
while_cond_11122998
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11122998___redundant_placeholder06
2while_while_cond_11122998___redundant_placeholder16
2while_while_cond_11122998___redundant_placeholder26
2while_while_cond_11122998___redundant_placeholder3
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
С6
┌
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11122554

inputs

states*
readvariableop_resource:	м1
matmul_readvariableop_resource:	м3
 matmul_1_readvariableop_resource:	dм
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:         R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:         O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:й
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>м
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:й
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>м
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:         g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	м*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:         u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d_
mul_1MulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         d[
add_2AddV2split:output:2	mul_1:z:0*
T0*'
_output_shapes
:         dI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_2MulSigmoid:y:0states*
T0*'
_output_shapes
:         dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         d[
mul_3Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:         dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         dЙ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         d
 
_user_specified_namestates
─

л
 Hidden_layer_while_cond_111233216
2hidden_layer_while_hidden_layer_while_loop_counter<
8hidden_layer_while_hidden_layer_while_maximum_iterations"
hidden_layer_while_placeholder$
 hidden_layer_while_placeholder_1$
 hidden_layer_while_placeholder_28
4hidden_layer_while_less_hidden_layer_strided_slice_1P
Lhidden_layer_while_hidden_layer_while_cond_11123321___redundant_placeholder0P
Lhidden_layer_while_hidden_layer_while_cond_11123321___redundant_placeholder1P
Lhidden_layer_while_hidden_layer_while_cond_11123321___redundant_placeholder2P
Lhidden_layer_while_hidden_layer_while_cond_11123321___redundant_placeholder3
hidden_layer_while_identity
Ц
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
ь
И
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123202
input_layer(
hidden_layer_11123189:	м(
hidden_layer_11123191:	м(
hidden_layer_11123193:	dм'
output_layer_11123196:d#
output_layer_11123198:
identityИв$Hidden_layer/StatefulPartitionedCallв$Output_layer/StatefulPartitionedCallд
$Hidden_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_11123189hidden_layer_11123191hidden_layer_11123193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11122846н
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall-Hidden_layer/StatefulPartitionedCall:output:0output_layer_11123196output_layer_11123198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_11122864|
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ф
NoOpNoOp%^Hidden_layer/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 2L
$Hidden_layer/StatefulPartitionedCall$Hidden_layer/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:X T
+
_output_shapes
:         d
%
_user_specified_nameinput_layer
╫}
Ш
#__inference__wrapped_model_11122309
input_layerO
<sequential_4_hidden_layer_gru_cell_4_readvariableop_resource:	мV
Csequential_4_hidden_layer_gru_cell_4_matmul_readvariableop_resource:	мX
Esequential_4_hidden_layer_gru_cell_4_matmul_1_readvariableop_resource:	dмJ
8sequential_4_output_layer_matmul_readvariableop_resource:dG
9sequential_4_output_layer_biasadd_readvariableop_resource:
identityИв:sequential_4/Hidden_layer/gru_cell_4/MatMul/ReadVariableOpв<sequential_4/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOpв3sequential_4/Hidden_layer/gru_cell_4/ReadVariableOpвsequential_4/Hidden_layer/whileв0sequential_4/Output_layer/BiasAdd/ReadVariableOpв/sequential_4/Output_layer/MatMul/ReadVariableOpZ
sequential_4/Hidden_layer/ShapeShapeinput_layer*
T0*
_output_shapes
:w
-sequential_4/Hidden_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_4/Hidden_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_4/Hidden_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'sequential_4/Hidden_layer/strided_sliceStridedSlice(sequential_4/Hidden_layer/Shape:output:06sequential_4/Hidden_layer/strided_slice/stack:output:08sequential_4/Hidden_layer/strided_slice/stack_1:output:08sequential_4/Hidden_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_4/Hidden_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d┴
&sequential_4/Hidden_layer/zeros/packedPack0sequential_4/Hidden_layer/strided_slice:output:01sequential_4/Hidden_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_4/Hidden_layer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ║
sequential_4/Hidden_layer/zerosFill/sequential_4/Hidden_layer/zeros/packed:output:0.sequential_4/Hidden_layer/zeros/Const:output:0*
T0*'
_output_shapes
:         d}
(sequential_4/Hidden_layer/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ж
#sequential_4/Hidden_layer/transpose	Transposeinput_layer1sequential_4/Hidden_layer/transpose/perm:output:0*
T0*+
_output_shapes
:d         x
!sequential_4/Hidden_layer/Shape_1Shape'sequential_4/Hidden_layer/transpose:y:0*
T0*
_output_shapes
:y
/sequential_4/Hidden_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/Hidden_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/Hidden_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▌
)sequential_4/Hidden_layer/strided_slice_1StridedSlice*sequential_4/Hidden_layer/Shape_1:output:08sequential_4/Hidden_layer/strided_slice_1/stack:output:0:sequential_4/Hidden_layer/strided_slice_1/stack_1:output:0:sequential_4/Hidden_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskА
5sequential_4/Hidden_layer/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         В
'sequential_4/Hidden_layer/TensorArrayV2TensorListReserve>sequential_4/Hidden_layer/TensorArrayV2/element_shape:output:02sequential_4/Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥а
Osequential_4/Hidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
Asequential_4/Hidden_layer/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_4/Hidden_layer/transpose:y:0Xsequential_4/Hidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥y
/sequential_4/Hidden_layer/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/Hidden_layer/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/Hidden_layer/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
)sequential_4/Hidden_layer/strided_slice_2StridedSlice'sequential_4/Hidden_layer/transpose:y:08sequential_4/Hidden_layer/strided_slice_2/stack:output:0:sequential_4/Hidden_layer/strided_slice_2/stack_1:output:0:sequential_4/Hidden_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЦ
4sequential_4/Hidden_layer/gru_cell_4/ones_like/ShapeShape2sequential_4/Hidden_layer/strided_slice_2:output:0*
T0*
_output_shapes
:y
4sequential_4/Hidden_layer/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ц
.sequential_4/Hidden_layer/gru_cell_4/ones_likeFill=sequential_4/Hidden_layer/gru_cell_4/ones_like/Shape:output:0=sequential_4/Hidden_layer/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         ▒
3sequential_4/Hidden_layer/gru_cell_4/ReadVariableOpReadVariableOp<sequential_4_hidden_layer_gru_cell_4_readvariableop_resource*
_output_shapes
:	м*
dtype0л
,sequential_4/Hidden_layer/gru_cell_4/unstackUnpack;sequential_4/Hidden_layer/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
num╬
(sequential_4/Hidden_layer/gru_cell_4/mulMul2sequential_4/Hidden_layer/strided_slice_2:output:07sequential_4/Hidden_layer/gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         ┐
:sequential_4/Hidden_layer/gru_cell_4/MatMul/ReadVariableOpReadVariableOpCsequential_4_hidden_layer_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype0┌
+sequential_4/Hidden_layer/gru_cell_4/MatMulMatMul,sequential_4/Hidden_layer/gru_cell_4/mul:z:0Bsequential_4/Hidden_layer/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м╪
,sequential_4/Hidden_layer/gru_cell_4/BiasAddBiasAdd5sequential_4/Hidden_layer/gru_cell_4/MatMul:product:05sequential_4/Hidden_layer/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         м
4sequential_4/Hidden_layer/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Т
*sequential_4/Hidden_layer/gru_cell_4/splitSplit=sequential_4/Hidden_layer/gru_cell_4/split/split_dim:output:05sequential_4/Hidden_layer/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_split├
<sequential_4/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpEsequential_4_hidden_layer_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0┌
-sequential_4/Hidden_layer/gru_cell_4/MatMul_1MatMul(sequential_4/Hidden_layer/zeros:output:0Dsequential_4/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м▄
.sequential_4/Hidden_layer/gru_cell_4/BiasAdd_1BiasAdd7sequential_4/Hidden_layer/gru_cell_4/MatMul_1:product:05sequential_4/Hidden_layer/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         м
*sequential_4/Hidden_layer/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       Б
6sequential_4/Hidden_layer/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┌
,sequential_4/Hidden_layer/gru_cell_4/split_1SplitV7sequential_4/Hidden_layer/gru_cell_4/BiasAdd_1:output:03sequential_4/Hidden_layer/gru_cell_4/Const:output:0?sequential_4/Hidden_layer/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_split╧
(sequential_4/Hidden_layer/gru_cell_4/addAddV23sequential_4/Hidden_layer/gru_cell_4/split:output:05sequential_4/Hidden_layer/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dЧ
,sequential_4/Hidden_layer/gru_cell_4/SigmoidSigmoid,sequential_4/Hidden_layer/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         d╤
*sequential_4/Hidden_layer/gru_cell_4/add_1AddV23sequential_4/Hidden_layer/gru_cell_4/split:output:15sequential_4/Hidden_layer/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dЫ
.sequential_4/Hidden_layer/gru_cell_4/Sigmoid_1Sigmoid.sequential_4/Hidden_layer/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         d╬
*sequential_4/Hidden_layer/gru_cell_4/mul_1Mul2sequential_4/Hidden_layer/gru_cell_4/Sigmoid_1:y:05sequential_4/Hidden_layer/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d╩
*sequential_4/Hidden_layer/gru_cell_4/add_2AddV23sequential_4/Hidden_layer/gru_cell_4/split:output:2.sequential_4/Hidden_layer/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dУ
)sequential_4/Hidden_layer/gru_cell_4/ReluRelu.sequential_4/Hidden_layer/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         d┐
*sequential_4/Hidden_layer/gru_cell_4/mul_2Mul0sequential_4/Hidden_layer/gru_cell_4/Sigmoid:y:0(sequential_4/Hidden_layer/zeros:output:0*
T0*'
_output_shapes
:         do
*sequential_4/Hidden_layer/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╚
(sequential_4/Hidden_layer/gru_cell_4/subSub3sequential_4/Hidden_layer/gru_cell_4/sub/x:output:00sequential_4/Hidden_layer/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d╩
*sequential_4/Hidden_layer/gru_cell_4/mul_3Mul,sequential_4/Hidden_layer/gru_cell_4/sub:z:07sequential_4/Hidden_layer/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         d┼
*sequential_4/Hidden_layer/gru_cell_4/add_3AddV2.sequential_4/Hidden_layer/gru_cell_4/mul_2:z:0.sequential_4/Hidden_layer/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dИ
7sequential_4/Hidden_layer/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   Ж
)sequential_4/Hidden_layer/TensorArrayV2_1TensorListReserve@sequential_4/Hidden_layer/TensorArrayV2_1/element_shape:output:02sequential_4/Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥`
sequential_4/Hidden_layer/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_4/Hidden_layer/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         n
,sequential_4/Hidden_layer/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : С
sequential_4/Hidden_layer/whileWhile5sequential_4/Hidden_layer/while/loop_counter:output:0;sequential_4/Hidden_layer/while/maximum_iterations:output:0'sequential_4/Hidden_layer/time:output:02sequential_4/Hidden_layer/TensorArrayV2_1:handle:0(sequential_4/Hidden_layer/zeros:output:02sequential_4/Hidden_layer/strided_slice_1:output:0Qsequential_4/Hidden_layer/TensorArrayUnstack/TensorListFromTensor:output_handle:0<sequential_4_hidden_layer_gru_cell_4_readvariableop_resourceCsequential_4_hidden_layer_gru_cell_4_matmul_readvariableop_resourceEsequential_4_hidden_layer_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_4_Hidden_layer_while_body_11122210*9
cond1R/
-sequential_4_Hidden_layer_while_cond_11122209*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations Ы
Jsequential_4/Hidden_layer/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   Р
<sequential_4/Hidden_layer/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_4/Hidden_layer/while:output:3Ssequential_4/Hidden_layer/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d         d*
element_dtype0В
/sequential_4/Hidden_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         {
1sequential_4/Hidden_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/Hidden_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
)sequential_4/Hidden_layer/strided_slice_3StridedSliceEsequential_4/Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:08sequential_4/Hidden_layer/strided_slice_3/stack:output:0:sequential_4/Hidden_layer/strided_slice_3/stack_1:output:0:sequential_4/Hidden_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_mask
*sequential_4/Hidden_layer/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ф
%sequential_4/Hidden_layer/transpose_1	TransposeEsequential_4/Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/Hidden_layer/transpose_1/perm:output:0*
T0*+
_output_shapes
:         ddu
!sequential_4/Hidden_layer/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    и
/sequential_4/Output_layer/MatMul/ReadVariableOpReadVariableOp8sequential_4_output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0╔
 sequential_4/Output_layer/MatMulMatMul2sequential_4/Hidden_layer/strided_slice_3:output:07sequential_4/Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ж
0sequential_4/Output_layer/BiasAdd/ReadVariableOpReadVariableOp9sequential_4_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0─
!sequential_4/Output_layer/BiasAddBiasAdd*sequential_4/Output_layer/MatMul:product:08sequential_4/Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         y
IdentityIdentity*sequential_4/Output_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:          
NoOpNoOp;^sequential_4/Hidden_layer/gru_cell_4/MatMul/ReadVariableOp=^sequential_4/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp4^sequential_4/Hidden_layer/gru_cell_4/ReadVariableOp ^sequential_4/Hidden_layer/while1^sequential_4/Output_layer/BiasAdd/ReadVariableOp0^sequential_4/Output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 2x
:sequential_4/Hidden_layer/gru_cell_4/MatMul/ReadVariableOp:sequential_4/Hidden_layer/gru_cell_4/MatMul/ReadVariableOp2|
<sequential_4/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp<sequential_4/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp2j
3sequential_4/Hidden_layer/gru_cell_4/ReadVariableOp3sequential_4/Hidden_layer/gru_cell_4/ReadVariableOp2B
sequential_4/Hidden_layer/whilesequential_4/Hidden_layer/while2d
0sequential_4/Output_layer/BiasAdd/ReadVariableOp0sequential_4/Output_layer/BiasAdd/ReadVariableOp2b
/sequential_4/Output_layer/MatMul/ReadVariableOp/sequential_4/Output_layer/MatMul/ReadVariableOp:X T
+
_output_shapes
:         d
%
_user_specified_nameinput_layer
ї@
Б
while_body_11123765
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	мD
1while_gru_cell_4_matmul_readvariableop_resource_0:	мF
3while_gru_cell_4_matmul_1_readvariableop_resource_0:	dм
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	мB
/while_gru_cell_4_matmul_readvariableop_resource:	мD
1while_gru_cell_4_matmul_1_readvariableop_resource:	dмИв&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0А
 while/gru_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:e
 while/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
while/gru_cell_4/ones_likeFill)while/gru_cell_4/ones_like/Shape:output:0)while/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numд
while/gru_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         Щ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Ю
while/gru_cell_4/MatMulMatMulwhile/gru_cell_4/mul:z:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мЬ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╓
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitЭ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	dм*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ма
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         К
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitУ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         do
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         dХ
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         ds
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dТ
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         dО
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dk
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dВ
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:         d[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         dО
while/gru_cell_4/mul_3Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dЙ
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_2:z:0while/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         d├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥w
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         d┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
Дk
К
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11123116

inputs5
"gru_cell_4_readvariableop_resource:	м<
)gru_cell_4_matmul_readvariableop_resource:	м>
+gru_cell_4_matmul_1_readvariableop_resource:	dм
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhile;
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
valueB:╤
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
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskb
gru_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:_
gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
gru_cell_4/ones_likeFill#gru_cell_4/ones_like/Shape:output:0#gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         ]
gru_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?С
gru_cell_4/dropout/MulMulgru_cell_4/ones_like:output:0!gru_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:         e
gru_cell_4/dropout/ShapeShapegru_cell_4/ones_like:output:0*
T0*
_output_shapes
:о
/gru_cell_4/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*f
!gru_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>╟
gru_cell_4/dropout/GreaterEqualGreaterEqual8gru_cell_4/dropout/random_uniform/RandomUniform:output:0*gru_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
gru_cell_4/dropout/CastCast#gru_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         К
gru_cell_4/dropout/Mul_1Mulgru_cell_4/dropout/Mul:z:0gru_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:         _
gru_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?Х
gru_cell_4/dropout_1/MulMulgru_cell_4/ones_like:output:0#gru_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:         g
gru_cell_4/dropout_1/ShapeShapegru_cell_4/ones_like:output:0*
T0*
_output_shapes
:┐
1gru_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2h
#gru_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>═
!gru_cell_4/dropout_1/GreaterEqualGreaterEqual:gru_cell_4/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Й
gru_cell_4/dropout_1/CastCast%gru_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         Р
gru_cell_4/dropout_1/Mul_1Mulgru_cell_4/dropout_1/Mul:z:0gru_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         _
gru_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?Х
gru_cell_4/dropout_2/MulMulgru_cell_4/ones_like:output:0#gru_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:         g
gru_cell_4/dropout_2/ShapeShapegru_cell_4/ones_like:output:0*
T0*
_output_shapes
:┐
1gru_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2h
#gru_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>═
!gru_cell_4/dropout_2/GreaterEqualGreaterEqual:gru_cell_4/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Й
gru_cell_4/dropout_2/CastCast%gru_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         Р
gru_cell_4/dropout_2/Mul_1Mulgru_cell_4/dropout_2/Mul:z:0gru_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         }
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	м*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
num
gru_cell_4/mulMulstrided_slice_2:output:0gru_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Л
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype0М
gru_cell_4/MatMulMatMulgru_cell_4/mul:z:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мК
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ─
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitП
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мО
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Є
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitБ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dc
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:         dГ
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dg
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dА
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d|
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         d_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dq
gru_cell_4/mul_2Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:         dU
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d|
gru_cell_4/mul_3Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dw
gru_cell_4/add_3AddV2gru_cell_4/mul_2:z:0gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┐
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11122999*
condR
while_cond_11122998*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d         d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         dd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         d: : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
·Й
л
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123636

inputsB
/hidden_layer_gru_cell_4_readvariableop_resource:	мI
6hidden_layer_gru_cell_4_matmul_readvariableop_resource:	мK
8hidden_layer_gru_cell_4_matmul_1_readvariableop_resource:	dм=
+output_layer_matmul_readvariableop_resource:d:
,output_layer_biasadd_readvariableop_resource:
identityИв-Hidden_layer/gru_cell_4/MatMul/ReadVariableOpв/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOpв&Hidden_layer/gru_cell_4/ReadVariableOpвHidden_layer/whileв#Output_layer/BiasAdd/ReadVariableOpв"Output_layer/MatMul/ReadVariableOpH
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
valueB:Т
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
value	B :dЪ
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
 *    У
Hidden_layer/zerosFill"Hidden_layer/zeros/packed:output:0!Hidden_layer/zeros/Const:output:0*
T0*'
_output_shapes
:         dp
Hidden_layer/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          З
Hidden_layer/transpose	Transposeinputs$Hidden_layer/transpose/perm:output:0*
T0*+
_output_shapes
:d         ^
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
valueB:Ь
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
         █
Hidden_layer/TensorArrayV2TensorListReserve1Hidden_layer/TensorArrayV2/element_shape:output:0%Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥У
BHidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       З
4Hidden_layer/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorHidden_layer/transpose:y:0KHidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥l
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
valueB:к
Hidden_layer/strided_slice_2StridedSliceHidden_layer/transpose:y:0+Hidden_layer/strided_slice_2/stack:output:0-Hidden_layer/strided_slice_2/stack_1:output:0-Hidden_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask|
'Hidden_layer/gru_cell_4/ones_like/ShapeShape%Hidden_layer/strided_slice_2:output:0*
T0*
_output_shapes
:l
'Hidden_layer/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┐
!Hidden_layer/gru_cell_4/ones_likeFill0Hidden_layer/gru_cell_4/ones_like/Shape:output:00Hidden_layer/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         j
%Hidden_layer/gru_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?╕
#Hidden_layer/gru_cell_4/dropout/MulMul*Hidden_layer/gru_cell_4/ones_like:output:0.Hidden_layer/gru_cell_4/dropout/Const:output:0*
T0*'
_output_shapes
:         
%Hidden_layer/gru_cell_4/dropout/ShapeShape*Hidden_layer/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:╚
<Hidden_layer/gru_cell_4/dropout/random_uniform/RandomUniformRandomUniform.Hidden_layer/gru_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*s
.Hidden_layer/gru_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>ю
,Hidden_layer/gru_cell_4/dropout/GreaterEqualGreaterEqualEHidden_layer/gru_cell_4/dropout/random_uniform/RandomUniform:output:07Hidden_layer/gru_cell_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Я
$Hidden_layer/gru_cell_4/dropout/CastCast0Hidden_layer/gru_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ▒
%Hidden_layer/gru_cell_4/dropout/Mul_1Mul'Hidden_layer/gru_cell_4/dropout/Mul:z:0(Hidden_layer/gru_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:         l
'Hidden_layer/gru_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?╝
%Hidden_layer/gru_cell_4/dropout_1/MulMul*Hidden_layer/gru_cell_4/ones_like:output:00Hidden_layer/gru_cell_4/dropout_1/Const:output:0*
T0*'
_output_shapes
:         Б
'Hidden_layer/gru_cell_4/dropout_1/ShapeShape*Hidden_layer/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:┘
>Hidden_layer/gru_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform0Hidden_layer/gru_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2u
0Hidden_layer/gru_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>Ї
.Hidden_layer/gru_cell_4/dropout_1/GreaterEqualGreaterEqualGHidden_layer/gru_cell_4/dropout_1/random_uniform/RandomUniform:output:09Hidden_layer/gru_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         г
&Hidden_layer/gru_cell_4/dropout_1/CastCast2Hidden_layer/gru_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ╖
'Hidden_layer/gru_cell_4/dropout_1/Mul_1Mul)Hidden_layer/gru_cell_4/dropout_1/Mul:z:0*Hidden_layer/gru_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         l
'Hidden_layer/gru_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?╝
%Hidden_layer/gru_cell_4/dropout_2/MulMul*Hidden_layer/gru_cell_4/ones_like:output:00Hidden_layer/gru_cell_4/dropout_2/Const:output:0*
T0*'
_output_shapes
:         Б
'Hidden_layer/gru_cell_4/dropout_2/ShapeShape*Hidden_layer/gru_cell_4/ones_like:output:0*
T0*
_output_shapes
:┘
>Hidden_layer/gru_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform0Hidden_layer/gru_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2u
0Hidden_layer/gru_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>Ї
.Hidden_layer/gru_cell_4/dropout_2/GreaterEqualGreaterEqualGHidden_layer/gru_cell_4/dropout_2/random_uniform/RandomUniform:output:09Hidden_layer/gru_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         г
&Hidden_layer/gru_cell_4/dropout_2/CastCast2Hidden_layer/gru_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ╖
'Hidden_layer/gru_cell_4/dropout_2/Mul_1Mul)Hidden_layer/gru_cell_4/dropout_2/Mul:z:0*Hidden_layer/gru_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         Ч
&Hidden_layer/gru_cell_4/ReadVariableOpReadVariableOp/hidden_layer_gru_cell_4_readvariableop_resource*
_output_shapes
:	м*
dtype0С
Hidden_layer/gru_cell_4/unstackUnpack.Hidden_layer/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numж
Hidden_layer/gru_cell_4/mulMul%Hidden_layer/strided_slice_2:output:0)Hidden_layer/gru_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         е
-Hidden_layer/gru_cell_4/MatMul/ReadVariableOpReadVariableOp6hidden_layer_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype0│
Hidden_layer/gru_cell_4/MatMulMatMulHidden_layer/gru_cell_4/mul:z:05Hidden_layer/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м▒
Hidden_layer/gru_cell_4/BiasAddBiasAdd(Hidden_layer/gru_cell_4/MatMul:product:0(Hidden_layer/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мr
'Hidden_layer/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ы
Hidden_layer/gru_cell_4/splitSplit0Hidden_layer/gru_cell_4/split/split_dim:output:0(Hidden_layer/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitй
/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp8hidden_layer_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0│
 Hidden_layer/gru_cell_4/MatMul_1MatMulHidden_layer/zeros:output:07Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м╡
!Hidden_layer/gru_cell_4/BiasAdd_1BiasAdd*Hidden_layer/gru_cell_4/MatMul_1:product:0(Hidden_layer/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мr
Hidden_layer/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       t
)Hidden_layer/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
Hidden_layer/gru_cell_4/split_1SplitV*Hidden_layer/gru_cell_4/BiasAdd_1:output:0&Hidden_layer/gru_cell_4/Const:output:02Hidden_layer/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitи
Hidden_layer/gru_cell_4/addAddV2&Hidden_layer/gru_cell_4/split:output:0(Hidden_layer/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         d}
Hidden_layer/gru_cell_4/SigmoidSigmoidHidden_layer/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         dк
Hidden_layer/gru_cell_4/add_1AddV2&Hidden_layer/gru_cell_4/split:output:1(Hidden_layer/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dБ
!Hidden_layer/gru_cell_4/Sigmoid_1Sigmoid!Hidden_layer/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dз
Hidden_layer/gru_cell_4/mul_1Mul%Hidden_layer/gru_cell_4/Sigmoid_1:y:0(Hidden_layer/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         dг
Hidden_layer/gru_cell_4/add_2AddV2&Hidden_layer/gru_cell_4/split:output:2!Hidden_layer/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dy
Hidden_layer/gru_cell_4/ReluRelu!Hidden_layer/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dШ
Hidden_layer/gru_cell_4/mul_2Mul#Hidden_layer/gru_cell_4/Sigmoid:y:0Hidden_layer/zeros:output:0*
T0*'
_output_shapes
:         db
Hidden_layer/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?б
Hidden_layer/gru_cell_4/subSub&Hidden_layer/gru_cell_4/sub/x:output:0#Hidden_layer/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         dг
Hidden_layer/gru_cell_4/mul_3MulHidden_layer/gru_cell_4/sub:z:0*Hidden_layer/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dЮ
Hidden_layer/gru_cell_4/add_3AddV2!Hidden_layer/gru_cell_4/mul_2:z:0!Hidden_layer/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         d{
*Hidden_layer/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ▀
Hidden_layer/TensorArrayV2_1TensorListReserve3Hidden_layer/TensorArrayV2_1/element_shape:output:0%Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥S
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
         a
Hidden_layer/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
Hidden_layer/whileWhile(Hidden_layer/while/loop_counter:output:0.Hidden_layer/while/maximum_iterations:output:0Hidden_layer/time:output:0%Hidden_layer/TensorArrayV2_1:handle:0Hidden_layer/zeros:output:0%Hidden_layer/strided_slice_1:output:0DHidden_layer/TensorArrayUnstack/TensorListFromTensor:output_handle:0/hidden_layer_gru_cell_4_readvariableop_resource6hidden_layer_gru_cell_4_matmul_readvariableop_resource8hidden_layer_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 Hidden_layer_while_body_11123513*,
cond$R"
 Hidden_layer_while_cond_11123512*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations О
=Hidden_layer/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   щ
/Hidden_layer/TensorArrayV2Stack/TensorListStackTensorListStackHidden_layer/while:output:3FHidden_layer/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d         d*
element_dtype0u
"Hidden_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         n
$Hidden_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$Hidden_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╚
Hidden_layer/strided_slice_3StridedSlice8Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:0+Hidden_layer/strided_slice_3/stack:output:0-Hidden_layer/strided_slice_3/stack_1:output:0-Hidden_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maskr
Hidden_layer/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╜
Hidden_layer/transpose_1	Transpose8Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:0&Hidden_layer/transpose_1/perm:output:0*
T0*+
_output_shapes
:         ddh
Hidden_layer/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    О
"Output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0в
Output_layer/MatMulMatMul%Hidden_layer/strided_slice_3:output:0*Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         М
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
Output_layer/BiasAddBiasAddOutput_layer/MatMul:product:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         l
IdentityIdentityOutput_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ▒
NoOpNoOp.^Hidden_layer/gru_cell_4/MatMul/ReadVariableOp0^Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp'^Hidden_layer/gru_cell_4/ReadVariableOp^Hidden_layer/while$^Output_layer/BiasAdd/ReadVariableOp#^Output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 2^
-Hidden_layer/gru_cell_4/MatMul/ReadVariableOp-Hidden_layer/gru_cell_4/MatMul/ReadVariableOp2b
/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp2P
&Hidden_layer/gru_cell_4/ReadVariableOp&Hidden_layer/gru_cell_4/ReadVariableOp2(
Hidden_layer/whileHidden_layer/while2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2H
"Output_layer/MatMul/ReadVariableOp"Output_layer/MatMul/ReadVariableOp:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
Й
└
/__inference_Hidden_layer_layer_call_fn_11123697

inputs
unknown:	м
	unknown_0:	м
	unknown_1:	dм
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11123116o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
Й
└
/__inference_Hidden_layer_layer_call_fn_11123686

inputs
unknown:	м
	unknown_0:	м
	unknown_1:	dм
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11122846o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
═	
√
J__inference_Output_layer_layer_call_and_return_conditional_losses_11122864

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ш 
╝
while_body_11122396
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_4_11122418_0:	м.
while_gru_cell_4_11122420_0:	м.
while_gru_cell_4_11122422_0:	dм
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_4_11122418:	м,
while_gru_cell_4_11122420:	м,
while_gru_cell_4_11122422:	dмИв(while/gru_cell_4/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0З
(while/gru_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_4_11122418_0while_gru_cell_4_11122420_0while_gru_cell_4_11122422_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11122383┌
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥О
while/Identity_4Identity1while/gru_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         dw

while/NoOpNoOp)^while/gru_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_4_11122418while_gru_cell_4_11122418_0"8
while_gru_cell_4_11122420while_gru_cell_4_11122420_0"8
while_gru_cell_4_11122422while_gru_cell_4_11122422_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2T
(while/gru_cell_4/StatefulPartitionedCall(while/gru_cell_4/StatefulPartitionedCall: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
ф
┤
while_cond_11122605
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11122605___redundant_placeholder06
2while_while_cond_11122605___redundant_placeholder16
2while_while_cond_11122605___redundant_placeholder26
2while_while_cond_11122605___redundant_placeholder3
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
║P
К
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11124228

inputs5
"gru_cell_4_readvariableop_resource:	м<
)gru_cell_4_matmul_readvariableop_resource:	м>
+gru_cell_4_matmul_1_readvariableop_resource:	dм
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhile;
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
valueB:╤
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
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskb
gru_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:_
gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
gru_cell_4/ones_likeFill#gru_cell_4/ones_like/Shape:output:0#gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         }
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	м*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numА
gru_cell_4/mulMulstrided_slice_2:output:0gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         Л
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype0М
gru_cell_4/MatMulMatMulgru_cell_4/mul:z:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мК
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ─
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitП
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мО
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Є
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitБ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dc
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:         dГ
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dg
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dА
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d|
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         d_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dq
gru_cell_4/mul_2Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:         dU
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d|
gru_cell_4/mul_3Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dw
gru_cell_4/add_3AddV2gru_cell_4/mul_2:z:0gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┐
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11124135*
condR
while_cond_11124134*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d         d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         dd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         d: : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
Цj
л
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123421

inputsB
/hidden_layer_gru_cell_4_readvariableop_resource:	мI
6hidden_layer_gru_cell_4_matmul_readvariableop_resource:	мK
8hidden_layer_gru_cell_4_matmul_1_readvariableop_resource:	dм=
+output_layer_matmul_readvariableop_resource:d:
,output_layer_biasadd_readvariableop_resource:
identityИв-Hidden_layer/gru_cell_4/MatMul/ReadVariableOpв/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOpв&Hidden_layer/gru_cell_4/ReadVariableOpвHidden_layer/whileв#Output_layer/BiasAdd/ReadVariableOpв"Output_layer/MatMul/ReadVariableOpH
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
valueB:Т
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
value	B :dЪ
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
 *    У
Hidden_layer/zerosFill"Hidden_layer/zeros/packed:output:0!Hidden_layer/zeros/Const:output:0*
T0*'
_output_shapes
:         dp
Hidden_layer/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          З
Hidden_layer/transpose	Transposeinputs$Hidden_layer/transpose/perm:output:0*
T0*+
_output_shapes
:d         ^
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
valueB:Ь
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
         █
Hidden_layer/TensorArrayV2TensorListReserve1Hidden_layer/TensorArrayV2/element_shape:output:0%Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥У
BHidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       З
4Hidden_layer/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorHidden_layer/transpose:y:0KHidden_layer/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥l
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
valueB:к
Hidden_layer/strided_slice_2StridedSliceHidden_layer/transpose:y:0+Hidden_layer/strided_slice_2/stack:output:0-Hidden_layer/strided_slice_2/stack_1:output:0-Hidden_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask|
'Hidden_layer/gru_cell_4/ones_like/ShapeShape%Hidden_layer/strided_slice_2:output:0*
T0*
_output_shapes
:l
'Hidden_layer/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┐
!Hidden_layer/gru_cell_4/ones_likeFill0Hidden_layer/gru_cell_4/ones_like/Shape:output:00Hidden_layer/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         Ч
&Hidden_layer/gru_cell_4/ReadVariableOpReadVariableOp/hidden_layer_gru_cell_4_readvariableop_resource*
_output_shapes
:	м*
dtype0С
Hidden_layer/gru_cell_4/unstackUnpack.Hidden_layer/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numз
Hidden_layer/gru_cell_4/mulMul%Hidden_layer/strided_slice_2:output:0*Hidden_layer/gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         е
-Hidden_layer/gru_cell_4/MatMul/ReadVariableOpReadVariableOp6hidden_layer_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype0│
Hidden_layer/gru_cell_4/MatMulMatMulHidden_layer/gru_cell_4/mul:z:05Hidden_layer/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м▒
Hidden_layer/gru_cell_4/BiasAddBiasAdd(Hidden_layer/gru_cell_4/MatMul:product:0(Hidden_layer/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мr
'Hidden_layer/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ы
Hidden_layer/gru_cell_4/splitSplit0Hidden_layer/gru_cell_4/split/split_dim:output:0(Hidden_layer/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitй
/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp8hidden_layer_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0│
 Hidden_layer/gru_cell_4/MatMul_1MatMulHidden_layer/zeros:output:07Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м╡
!Hidden_layer/gru_cell_4/BiasAdd_1BiasAdd*Hidden_layer/gru_cell_4/MatMul_1:product:0(Hidden_layer/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мr
Hidden_layer/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       t
)Hidden_layer/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
Hidden_layer/gru_cell_4/split_1SplitV*Hidden_layer/gru_cell_4/BiasAdd_1:output:0&Hidden_layer/gru_cell_4/Const:output:02Hidden_layer/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitи
Hidden_layer/gru_cell_4/addAddV2&Hidden_layer/gru_cell_4/split:output:0(Hidden_layer/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         d}
Hidden_layer/gru_cell_4/SigmoidSigmoidHidden_layer/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         dк
Hidden_layer/gru_cell_4/add_1AddV2&Hidden_layer/gru_cell_4/split:output:1(Hidden_layer/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dБ
!Hidden_layer/gru_cell_4/Sigmoid_1Sigmoid!Hidden_layer/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dз
Hidden_layer/gru_cell_4/mul_1Mul%Hidden_layer/gru_cell_4/Sigmoid_1:y:0(Hidden_layer/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         dг
Hidden_layer/gru_cell_4/add_2AddV2&Hidden_layer/gru_cell_4/split:output:2!Hidden_layer/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dy
Hidden_layer/gru_cell_4/ReluRelu!Hidden_layer/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dШ
Hidden_layer/gru_cell_4/mul_2Mul#Hidden_layer/gru_cell_4/Sigmoid:y:0Hidden_layer/zeros:output:0*
T0*'
_output_shapes
:         db
Hidden_layer/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?б
Hidden_layer/gru_cell_4/subSub&Hidden_layer/gru_cell_4/sub/x:output:0#Hidden_layer/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         dг
Hidden_layer/gru_cell_4/mul_3MulHidden_layer/gru_cell_4/sub:z:0*Hidden_layer/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dЮ
Hidden_layer/gru_cell_4/add_3AddV2!Hidden_layer/gru_cell_4/mul_2:z:0!Hidden_layer/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         d{
*Hidden_layer/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ▀
Hidden_layer/TensorArrayV2_1TensorListReserve3Hidden_layer/TensorArrayV2_1/element_shape:output:0%Hidden_layer/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥S
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
         a
Hidden_layer/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
Hidden_layer/whileWhile(Hidden_layer/while/loop_counter:output:0.Hidden_layer/while/maximum_iterations:output:0Hidden_layer/time:output:0%Hidden_layer/TensorArrayV2_1:handle:0Hidden_layer/zeros:output:0%Hidden_layer/strided_slice_1:output:0DHidden_layer/TensorArrayUnstack/TensorListFromTensor:output_handle:0/hidden_layer_gru_cell_4_readvariableop_resource6hidden_layer_gru_cell_4_matmul_readvariableop_resource8hidden_layer_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 Hidden_layer_while_body_11123322*,
cond$R"
 Hidden_layer_while_cond_11123321*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations О
=Hidden_layer/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   щ
/Hidden_layer/TensorArrayV2Stack/TensorListStackTensorListStackHidden_layer/while:output:3FHidden_layer/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d         d*
element_dtype0u
"Hidden_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         n
$Hidden_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$Hidden_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╚
Hidden_layer/strided_slice_3StridedSlice8Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:0+Hidden_layer/strided_slice_3/stack:output:0-Hidden_layer/strided_slice_3/stack_1:output:0-Hidden_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maskr
Hidden_layer/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╜
Hidden_layer/transpose_1	Transpose8Hidden_layer/TensorArrayV2Stack/TensorListStack:tensor:0&Hidden_layer/transpose_1/perm:output:0*
T0*+
_output_shapes
:         ddh
Hidden_layer/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    О
"Output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0в
Output_layer/MatMulMatMul%Hidden_layer/strided_slice_3:output:0*Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         М
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
Output_layer/BiasAddBiasAddOutput_layer/MatMul:product:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         l
IdentityIdentityOutput_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ▒
NoOpNoOp.^Hidden_layer/gru_cell_4/MatMul/ReadVariableOp0^Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp'^Hidden_layer/gru_cell_4/ReadVariableOp^Hidden_layer/while$^Output_layer/BiasAdd/ReadVariableOp#^Output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 2^
-Hidden_layer/gru_cell_4/MatMul/ReadVariableOp-Hidden_layer/gru_cell_4/MatMul/ReadVariableOp2b
/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp/Hidden_layer/gru_cell_4/MatMul_1/ReadVariableOp2P
&Hidden_layer/gru_cell_4/ReadVariableOp&Hidden_layer/gru_cell_4/ReadVariableOp2(
Hidden_layer/whileHidden_layer/while2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2H
"Output_layer/MatMul/ReadVariableOp"Output_layer/MatMul/ReadVariableOp:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
║P
К
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11122846

inputs5
"gru_cell_4_readvariableop_resource:	м<
)gru_cell_4_matmul_readvariableop_resource:	м>
+gru_cell_4_matmul_1_readvariableop_resource:	dм
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhile;
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
valueB:╤
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
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskb
gru_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:_
gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
gru_cell_4/ones_likeFill#gru_cell_4/ones_like/Shape:output:0#gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         }
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	м*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numА
gru_cell_4/mulMulstrided_slice_2:output:0gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         Л
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype0М
gru_cell_4/MatMulMatMulgru_cell_4/mul:z:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мК
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ─
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitП
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мО
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Є
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitБ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dc
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:         dГ
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dg
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dА
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d|
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         d_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dq
gru_cell_4/mul_2Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:         dU
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d|
gru_cell_4/mul_3Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dw
gru_cell_4/add_3AddV2gru_cell_4/mul_2:z:0gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┐
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11122753*
condR
while_cond_11122752*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:d         d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         dd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         d: : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
яP
М
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11123858
inputs_05
"gru_cell_4_readvariableop_resource:	м<
)gru_cell_4_matmul_readvariableop_resource:	м>
+gru_cell_4_matmul_1_readvariableop_resource:	dм
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhile=
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
valueB:╤
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
:         dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskb
gru_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:_
gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
gru_cell_4/ones_likeFill#gru_cell_4/ones_like/Shape:output:0#gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         }
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	м*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numА
gru_cell_4/mulMulstrided_slice_2:output:0gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         Л
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype0М
gru_cell_4/MatMulMatMulgru_cell_4/mul:z:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мК
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ─
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitП
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мО
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Є
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitБ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         dc
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:         dГ
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         dg
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dА
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         d|
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         d_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dq
gru_cell_4/mul_2Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:         dU
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         d|
gru_cell_4/mul_3Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dw
gru_cell_4/add_3AddV2gru_cell_4/mul_2:z:0gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┐
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11123765*
condR
while_cond_11123764*8
output_shapes'
%: : : : :         d: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    d   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         d▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
╢

▄
-__inference_gru_cell_4_layer_call_fn_11124484

inputs
states_0
unknown:	м
	unknown_0:	м
	unknown_1:	dм
identity

identity_1ИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         d:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11122554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         d
"
_user_specified_name
states/0
г
в
-sequential_4_Hidden_layer_while_cond_11122209P
Lsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_loop_counterV
Rsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_maximum_iterations/
+sequential_4_hidden_layer_while_placeholder1
-sequential_4_hidden_layer_while_placeholder_11
-sequential_4_hidden_layer_while_placeholder_2R
Nsequential_4_hidden_layer_while_less_sequential_4_hidden_layer_strided_slice_1j
fsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_cond_11122209___redundant_placeholder0j
fsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_cond_11122209___redundant_placeholder1j
fsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_cond_11122209___redundant_placeholder2j
fsequential_4_hidden_layer_while_sequential_4_hidden_layer_while_cond_11122209___redundant_placeholder3,
(sequential_4_hidden_layer_while_identity
╩
$sequential_4/Hidden_layer/while/LessLess+sequential_4_hidden_layer_while_placeholderNsequential_4_hidden_layer_while_less_sequential_4_hidden_layer_strided_slice_1*
T0*
_output_shapes
: 
(sequential_4/Hidden_layer/while/IdentityIdentity(sequential_4/Hidden_layer/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_4_hidden_layer_while_identity1sequential_4/Hidden_layer/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
ф
┤
while_cond_11123764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11123764___redundant_placeholder06
2while_while_cond_11123764___redundant_placeholder16
2while_while_cond_11123764___redundant_placeholder26
2while_while_cond_11123764___redundant_placeholder3
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
-: : : : :         d: ::::: 
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
:         d:

_output_shapes
: :

_output_shapes
:
ї@
Б
while_body_11124135
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	мD
1while_gru_cell_4_matmul_readvariableop_resource_0:	мF
3while_gru_cell_4_matmul_1_readvariableop_resource_0:	dм
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	мB
/while_gru_cell_4_matmul_readvariableop_resource:	мD
1while_gru_cell_4_matmul_1_readvariableop_resource:	dмИв&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0А
 while/gru_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:e
 while/gru_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
while/gru_cell_4/ones_likeFill)while/gru_cell_4/ones_like/Shape:output:0)while/gru_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:         Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numд
while/gru_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_4/ones_like:output:0*
T0*'
_output_shapes
:         Щ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	м*
dtype0Ю
while/gru_cell_4/MatMulMatMulwhile/gru_cell_4/mul:z:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мЬ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         мk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╓
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splitЭ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	dм*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ма
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         мk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         К
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_splitУ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:         do
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:         dХ
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:         ds
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:         dТ
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:         dО
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         dk
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:         dВ
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:         d[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:         dО
while/gru_cell_4/mul_3Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         dЙ
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_2:z:0while/gru_cell_4/mul_3:z:0*
T0*'
_output_shapes
:         d├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥w
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         d┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         d: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 
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
:         d:

_output_shapes
: :

_output_shapes
: 
╝
Є
&__inference_signature_wrapper_11123653
input_layer
unknown:	м
	unknown_0:	м
	unknown_1:	dм
	unknown_2:d
	unknown_3:
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_11122309o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:         d
%
_user_specified_nameinput_layer
ў8
▒
!__inference__traced_save_11124689
file_prefix2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop=
9savev2_hidden_layer_gru_cell_4_kernel_read_readvariableopG
Csavev2_hidden_layer_gru_cell_4_recurrent_kernel_read_readvariableop;
7savev2_hidden_layer_gru_cell_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableopD
@savev2_adam_hidden_layer_gru_cell_4_kernel_m_read_readvariableopN
Jsavev2_adam_hidden_layer_gru_cell_4_recurrent_kernel_m_read_readvariableopB
>savev2_adam_hidden_layer_gru_cell_4_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableopD
@savev2_adam_hidden_layer_gru_cell_4_kernel_v_read_readvariableopN
Jsavev2_adam_hidden_layer_gru_cell_4_recurrent_kernel_v_read_readvariableopB
>savev2_adam_hidden_layer_gru_cell_4_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: С
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*║
value░BнB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЯ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B л
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop9savev2_hidden_layer_gru_cell_4_kernel_read_readvariableopCsavev2_hidden_layer_gru_cell_4_recurrent_kernel_read_readvariableop7savev2_hidden_layer_gru_cell_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop@savev2_adam_hidden_layer_gru_cell_4_kernel_m_read_readvariableopJsavev2_adam_hidden_layer_gru_cell_4_recurrent_kernel_m_read_readvariableop>savev2_adam_hidden_layer_gru_cell_4_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableop@savev2_adam_hidden_layer_gru_cell_4_kernel_v_read_readvariableopJsavev2_adam_hidden_layer_gru_cell_4_recurrent_kernel_v_read_readvariableop>savev2_adam_hidden_layer_gru_cell_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*╛
_input_shapesм
й: :d:: : : : : :	м:	dм:	м: : : : :d::	м:	dм:	м:d::	м:	dм:	м: 2(
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
:	м:%	!

_output_shapes
:	dм:%
!

_output_shapes
:	м:
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
:	м:%!

_output_shapes
:	dм:%!

_output_shapes
:	м:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	м:%!

_output_shapes
:	dм:%!

_output_shapes
:	м:

_output_shapes
: 
б
┬
/__inference_Hidden_layer_layer_call_fn_11123664
inputs_0
unknown:	м
	unknown_0:	м
	unknown_1:	dм
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11122460o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
ь
И
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123218
input_layer(
hidden_layer_11123205:	м(
hidden_layer_11123207:	м(
hidden_layer_11123209:	dм'
output_layer_11123212:d#
output_layer_11123214:
identityИв$Hidden_layer/StatefulPartitionedCallв$Output_layer/StatefulPartitionedCallд
$Hidden_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_11123205hidden_layer_11123207hidden_layer_11123209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11123116н
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall-Hidden_layer/StatefulPartitionedCall:output:0output_layer_11123212output_layer_11123214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_Output_layer_layer_call_and_return_conditional_losses_11122864|
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ф
NoOpNoOp%^Hidden_layer/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         d: : : : : 2L
$Hidden_layer/StatefulPartitionedCall$Hidden_layer/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:X T
+
_output_shapes
:         d
%
_user_specified_nameinput_layer
Щ6
▄
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11124594

inputs
states_0*
readvariableop_resource:	м1
matmul_readvariableop_resource:	м3
 matmul_1_readvariableop_resource:	dм
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:         R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:         O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:й
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>м
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:й
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*

seed**
seed2]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>м
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:         g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	м*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:м:м*	
numW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:         u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype0k
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:         d:         d:         d*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	dм*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:         d:         d:         d*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d_
mul_1MulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         d[
add_2AddV2split:output:2	mul_1:z:0*
T0*'
_output_shapes
:         dI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:         dU
mul_2MulSigmoid:y:0states_0*
T0*'
_output_shapes
:         dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         d[
mul_3Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:         dX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         dZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         dЙ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         d: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         d
"
_user_specified_name
states/0"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╗
serving_defaultз
G
input_layer8
serving_default_input_layer:0         d@
Output_layer0
StatefulPartitionedCall:0         tensorflow/serving/predict:дe
┤
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
┌
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
╗

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
н
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
╩
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
К2З
/__inference_sequential_4_layer_call_fn_11122884
/__inference_sequential_4_layer_call_fn_11123239
/__inference_sequential_4_layer_call_fn_11123254
/__inference_sequential_4_layer_call_fn_11123186└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ў2є
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123421
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123636
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123202
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123218└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥B╧
#__inference__wrapped_model_11122309input_layer"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,
*serving_default"
signature_map
ш

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
╣

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
Я2Ь
/__inference_Hidden_layer_layer_call_fn_11123664
/__inference_Hidden_layer_layer_call_fn_11123675
/__inference_Hidden_layer_layer_call_fn_11123686
/__inference_Hidden_layer_layer_call_fn_11123697╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Л2И
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11123858
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11124067
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11124228
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11124437╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
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
н
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
┘2╓
/__inference_Output_layer_layer_call_fn_11124446в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_Output_layer_layer_call_and_return_conditional_losses_11124456в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
1:/	м2Hidden_layer/gru_cell_4/kernel
;:9	dм2(Hidden_layer/gru_cell_4/recurrent_kernel
/:-	м2Hidden_layer/gru_cell_4/bias
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
╤B╬
&__inference_signature_wrapper_11123653input_layer"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
н
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
в2Я
-__inference_gru_cell_4_layer_call_fn_11124470
-__inference_gru_cell_4_layer_call_fn_11124484╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╪2╒
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11124527
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11124594╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
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
6:4	м2%Adam/Hidden_layer/gru_cell_4/kernel/m
@:>	dм2/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/m
4:2	м2#Adam/Hidden_layer/gru_cell_4/bias/m
*:(d2Adam/Output_layer/kernel/v
$:"2Adam/Output_layer/bias/v
6:4	м2%Adam/Hidden_layer/gru_cell_4/kernel/v
@:>	dм2/Adam/Hidden_layer/gru_cell_4/recurrent_kernel/v
4:2	м2#Adam/Hidden_layer/gru_cell_4/bias/v╦
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11123858}$"#OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "%в"
К
0         d
Ъ ╦
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11124067}$"#OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "%в"
К
0         d
Ъ ╗
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11124228m$"#?в<
5в2
$К!
inputs         d

 
p 

 
к "%в"
К
0         d
Ъ ╗
J__inference_Hidden_layer_layer_call_and_return_conditional_losses_11124437m$"#?в<
5в2
$К!
inputs         d

 
p

 
к "%в"
К
0         d
Ъ г
/__inference_Hidden_layer_layer_call_fn_11123664p$"#OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "К         dг
/__inference_Hidden_layer_layer_call_fn_11123675p$"#OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "К         dУ
/__inference_Hidden_layer_layer_call_fn_11123686`$"#?в<
5в2
$К!
inputs         d

 
p 

 
к "К         dУ
/__inference_Hidden_layer_layer_call_fn_11123697`$"#?в<
5в2
$К!
inputs         d

 
p

 
к "К         dк
J__inference_Output_layer_layer_call_and_return_conditional_losses_11124456\/в,
%в"
 К
inputs         d
к "%в"
К
0         
Ъ В
/__inference_Output_layer_layer_call_fn_11124446O/в,
%в"
 К
inputs         d
к "К         е
#__inference__wrapped_model_11122309~$"#8в5
.в+
)К&
input_layer         d
к ";к8
6
Output_layer&К#
Output_layer         Д
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11124527╖$"#\вY
RвO
 К
inputs         
'в$
"К
states/0         d
p 
к "RвO
HвE
К
0/0         d
$Ъ!
К
0/1/0         d
Ъ Д
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_11124594╖$"#\вY
RвO
 К
inputs         
'в$
"К
states/0         d
p
к "RвO
HвE
К
0/0         d
$Ъ!
К
0/1/0         d
Ъ █
-__inference_gru_cell_4_layer_call_fn_11124470й$"#\вY
RвO
 К
inputs         
'в$
"К
states/0         d
p 
к "DвA
К
0         d
"Ъ
К
1/0         d█
-__inference_gru_cell_4_layer_call_fn_11124484й$"#\вY
RвO
 К
inputs         
'в$
"К
states/0         d
p
к "DвA
К
0         d
"Ъ
К
1/0         d╛
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123202p$"#@в=
6в3
)К&
input_layer         d
p 

 
к "%в"
К
0         
Ъ ╛
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123218p$"#@в=
6в3
)К&
input_layer         d
p

 
к "%в"
К
0         
Ъ ╣
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123421k$"#;в8
1в.
$К!
inputs         d
p 

 
к "%в"
К
0         
Ъ ╣
J__inference_sequential_4_layer_call_and_return_conditional_losses_11123636k$"#;в8
1в.
$К!
inputs         d
p

 
к "%в"
К
0         
Ъ Ц
/__inference_sequential_4_layer_call_fn_11122884c$"#@в=
6в3
)К&
input_layer         d
p 

 
к "К         Ц
/__inference_sequential_4_layer_call_fn_11123186c$"#@в=
6в3
)К&
input_layer         d
p

 
к "К         С
/__inference_sequential_4_layer_call_fn_11123239^$"#;в8
1в.
$К!
inputs         d
p 

 
к "К         С
/__inference_sequential_4_layer_call_fn_11123254^$"#;в8
1в.
$К!
inputs         d
p

 
к "К         ╕
&__inference_signature_wrapper_11123653Н$"#GвD
в 
=к:
8
input_layer)К&
input_layer         d";к8
6
Output_layer&К#
Output_layer         