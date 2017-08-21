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
      std::cout << "HELLO I AM ARM ENgine" << std::endl;
      auto& X = Input(0);
      auto& filter = Input(1);
      auto& bias = Input(2);
      auto* Y = Output(0);
 
      //unsigned int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
      unsigned int N = 1, C = 1, H = 32, W = 32;
      unsigned int M = filter.dim32(0);  
      arm_compute::NEConvolutionLayer conv0;
      arm_compute::Tensor src;
      arm_compute::Tensor weights0;
      arm_compute::Tensor biases0;
      arm_compute::Tensor out_conv0;
      
      const arm_compute::TensorShape src_shape(W, H, N);
      std::cout << "ifm_src " << N << std::endl;
      src.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
      
      //ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
      //unsigned int oH = Y->dim32(2), oW = Y->dim32(3);
      unsigned int oH = 28, oW = 28;

      //unsigned int kernel_x_conv0 = filter.dim32(2);
      //unsigned int kernel_y_conv0 = filter.dim32(3);
      //unsigned int ofm_conv0 = M; 
      unsigned int kernel_x_conv0 = 5;
      unsigned int kernel_y_conv0 = 5;
      unsigned int ofm_conv0 = 8;
      const arm_compute::TensorShape weight_shape(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);

      const arm_compute::TensorShape biases_shape(weight_shape[3]);
      const arm_compute::TensorShape out_shape(oW, oH, weight_shape[3]);
      weights0.allocator()->init(arm_compute::TensorInfo(weight_shape, 1, arm_compute::DataType::F32));
      biases0.allocator()->init(arm_compute::TensorInfo(biases_shape, 1, arm_compute::DataType::F32));
      out_conv0.allocator()->init(arm_compute::TensorInfo(out_shape, 1, arm_compute::DataType::F32)); 

      conv0.configure(&src, &weights0, &biases0, &out_conv0, arm_compute::PadStrideInfo());
      std::cout << "start get x data" << std::endl;
      const uint8_t *_buf = X.template data<uint8_t>();
      std::cout << "end get x data size is : " << X.size() << std::endl;
      std::vector<uint8_t> buf(_buf, _buf + X.size());

      src.allocator()->allocate(buf);
      /*uint8_t *_res = src.allocator()->data();
      int len = src.allocator()->length();
      std::vector<uint8_t> res(_res, _res + len);
      for (int i = 0; i < res.size(); i++) {
	      std::cout << unsigned(res[i]) << " ";
      }*/
      std::cout << "start get weight data" << std::endl;
      const uint8_t *_weights = filter.template data<uint8_t>();
      std::vector<uint8_t> weights(_weights, _weights + filter.size());
      weights0.allocator()->allocate(weights);

      std::cout << "start get biases data" << std::endl;
      const uint8_t *_bias = bias.template data<uint8_t>();
      std::vector<uint8_t> biass(_bias, _bias + bias.size());
      biases0.allocator()->allocate(biass);

      out_conv0.allocator()->allocate();
      conv0.run();
      uint8_t *_res = weights0.allocator()->data();
      int len = weights0.allocator()->length();
      std::cout << "length of weight: " << len << std::endl;
      for (int i = 0; i < len; i++) {
	      std::cout << unsigned(*_res) << " ";
	      _res++;
      }
      uint8_t *_resa = out_conv0.allocator()->data();
      std::vector<unsigned int> y_vec{oW, oH, weight_shape[3]};
      Y->Resize(y_vec);
      uint8_t *_result = Y->template mutable_data<uint8_t>();
      _result = _resa;
      
      len = out_conv0.allocator()->length();
      std::cout << "length of weight: " << len << std::endl;
      for (int i = 0; i < len; i++) {
	      std::cout << unsigned(*_resa) << " ";
	      _resa++;
      }
      //uint8_t *_res = Y->template mutable_data<uint8_t>(); 
      /*uint8_t *_res = out_conv0.allocator()->data();
      int len = out_conv0.allocator()->length();
      for (int i = 0; i < len; i++) {
	      std::cout << unsigned(*_res) << " ";
	      _res++;
      }*/
      std::cout << std::endl;
    }
  private:
  };
  REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, ARM, ARMConvOp);
}
