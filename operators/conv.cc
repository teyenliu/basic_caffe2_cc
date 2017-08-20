#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/leaky_relu_op.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"


namespace caffe2 {
  class ARMConvOp final : public ConvPoolOpBase<CPUContext> {
  public:
    ARMConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {}
    bool RunOnDeviceWithOrderNCHW() override {
      auto& X = Input(0);
      auto& filter = Input(1);
      auto& bias = Input(2);
      auto* Y = Output(0);
 
      unsigned int C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
      unsigned int M = filter.dim32(0);  
      arm_compute::NEConvolutionLayer conv0;
      arm_compute::Tensor src;
      arm_compute::Tensor weights0;
      arm_compute::Tensor biases0;
      arm_compute::Tensor out_conv0;
      
      unsigned int kernel_x_conv0 = filter.dim32(2);
      unsigned int kernel_y_conv0 = filter.dim32(3);
      unsigned int ofm_conv0 = M; 
      const arm_compute::TensorShape src_shape(W, H, C);
      src.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
      
      ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
      unsigned int oH = Y->dim32(2), oW = Y->dim32(3);
      const arm_compute::TensorShape weight_shape(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);

      const arm_compute::TensorShape biases_shape(weight_shape[3]);
      const arm_compute::TensorShape out_shape(oW, oH, weight_shape[3]);
      weights0.allocator()->init(arm_compute::TensorInfo(weight_shape, 1, arm_compute::DataType::F32));
      biases0.allocator()->init(arm_compute::TensorInfo(biases_shape, 1, arm_compute::DataType::F32));
      out_conv0.allocator()->init(arm_compute::TensorInfo(out_shape, 1, arm_compute::DataType::F32)); 

      conv0.configure(&src, &weights0, &biases0, &out_conv0, arm_compute::PadStrideInfo());
      src.allocator()->allocate();
      weights0.allocator()->allocate();
      biases0.allocator()->allocate();
      out_conv0.allocator()->allocate();
      conv0.run();
    }
  private:
  };
  REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, ARM, ARMConvOp);
}
