#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/leaky_relu_op.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"

using namespace arm_compute;
namespace caffe2 {
  class ARMConvOp final : public ConvPoolOpBase<CPUContext> {
  public:
    ARMConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolBase<CPUContext>(oeprator_def, ws) {}
    bool RunOnDeviceWithOrderNCHW() override {
      auto& X = Input(0);
      auto& filter = Input(1);
      auto& bias = Input(2);
      auto* Y = Output(0);
 
      const int C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
      const int M = filter.dim32(0);  
      NEConvolutionLayer conv0;
      Tensor src;
      Tensor weights0;
      Tensor biases0;
      Tensor out_conv0;
      
      constexpr unsigned int kernel_x_conv0 = filter.dim32(2);
      constexpr unsigned int kernel_y_conv0 = filter.dim32(3);
      constexpr unsigned int ofm_conv0 = M; 
      const TensorShape src_shape(W, H, C);
      src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32);
      
      ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
      const int oH = Y->dim32(2), ow = Y->dim32(3);
      const TensofrShape weight_shape(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);

      const TensorShape biases_shape(weight_shape[3]);
      const TensorShape out_shape(oW, oH, weight_shape[3]);
      weight0.allocator()->init(TensorInfo(weight_shape, 1, DataType::F32);
      biases0.allocator()->init(TensorInfo(biases_shape, 1, DataType::F32);
      out_conv0.allocator()->init(TensorInfo(out_shape, 1, DataType::F32); 

      conv0.configure(&src, &weight0, &biases0, &out_conv0, PadStrideInfo());
      src.allocator()->allocate();
      weight0.allocator()->allocate();
      biases0.allocator()->allocate();
      out_conv0.allocator()->allocate();
      conv0.run();
    }
  private:
  };
  REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, ARM, ARMConvOp);
}
