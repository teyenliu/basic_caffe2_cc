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
    bool convertinput(std::vector<uint8_t>& dst, const uint8_t* src, int N, int C, int H, int W) {
      for (unsigned int i = 0; i < N; i++) {
	for (unsigned int j = 0; j < C; j++) {
          for (unsigned int k = 0; k < H; k++) {
	    for (unsigned int h = 0; h < W; h++) {
              dst[h * H * C + k * C + j] = src[i * C * H * W + j * W * H + k * W + h];
	    }
	  }
        }
      }
      std::cout << "finish input converter " << std::endl;
    }
    bool convertweight(std::vector<uint8_t>& dst, const uint8_t* src, int out, int in, int kernel) {
      for (uint8_t i = 0; i < out; i++) {
        for (uint8_t j = 0; j < in; j++) {
          for (uint8_t k = 0; k < kernel; k++) {
            for (uint8_t h = 0; h < kernel; h++) {
	       dst[h * kernel * in * out + k * in * out + j * out + i] = src[i * in * kernel * kernel + j * kernel * kernel + k * kernel + h];
	    }
	  }
	}
      }
      std::cout << "finish weight converter " << std::endl;
    }
    bool convertres(uint8_t* src, uint8_t* dst, int N, int C, int H, int W) {
      for (unsigned i = 0; i < N; i++) {
	for (unsigned int j = 0; j < C; j++) {
          for (unsigned int k = 0; k < H; k++) {
	    for (unsigned int h = 0; h < W; h++) {
            dst[i * H * W * C+ j * H * W + k * W + h] = src[h * H * C + k * C + j];
	  }
	}
	}
      }
    }
    bool RunOnDeviceWithOrderNCHW() override {
      std::cout << "HELLO I AM ARM ENgine" << std::endl;
      auto& X = Input(0);
      auto& filter = Input(1);
      auto& bias = Input(2);
      auto* Y = Output(0);
 
      unsigned int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
      //unsigned int N = 2, H = 32, W = 32;
      unsigned int M = filter.dim32(0);  
      arm_compute::NEConvolutionLayer conv0;
      arm_compute::Tensor src;
      arm_compute::Tensor weights0;
      arm_compute::Tensor biases0;
      arm_compute::Tensor out_conv0;
      
      const arm_compute::TensorShape src_shape(W, H, C);
      src.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
      
      ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
      unsigned int oH = Y->dim32(2), oW = Y->dim32(3);
      //unsigned int oH = 28, oW = 28;

      unsigned int kernel_x_conv0 = filter.dim32(2);
      unsigned int kernel_y_conv0 = filter.dim32(3);
      unsigned int ofm_conv0 = M; 
      //unsigned int kernel_x_conv0 = 5;
      //unsigned int kernel_y_conv0 = 5;
      //unsigned int ofm_conv0 = 8;
      const arm_compute::TensorShape weight_shape(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);

      const arm_compute::TensorShape biases_shape(weight_shape[3]);
      const arm_compute::TensorShape out_shape(oW, oH, weight_shape[3]);
      weights0.allocator()->init(arm_compute::TensorInfo(weight_shape, 1, arm_compute::DataType::F32));
      biases0.allocator()->init(arm_compute::TensorInfo(biases_shape, 1, arm_compute::DataType::F32));
      out_conv0.allocator()->init(arm_compute::TensorInfo(out_shape, 1, arm_compute::DataType::F32)); 

      conv0.configure(&src, &weights0, &biases0, &out_conv0, arm_compute::PadStrideInfo());
      std::cout << "start get x data" << std::endl;
      const uint8_t *_buf = X.template data<uint8_t>();
      std::vector<uint8_t> buf(X.size(), 0);
      if (X.dim32(1) == src_shape.z() && X.dim32(2) == src_shape.y() && X.dim32(3) == src_shape.x()) {
      convertinput(buf, _buf, N, C, H, W);
      src.allocator()->allocate(buf);
      }

      std::cout << "end get x data size is : " << X.size() << std::endl;
      /*uint8_t *_res = src.allocator()->data();
      int len = src.allocator()->length();
      std::vector<uint8_t> res(_res, _res + len);
      for (int i = 0; i < res.size(); i++) {
	      std::cout << unsigned(res[i]) << " ";
      }*/
      std::cout << "start get weight data" << std::endl;
      const uint8_t *_weights = filter.template data<uint8_t>();
      std::vector<uint8_t> weights(filter.size(), 0);
      std::cout << filter.dim32(0) << " " << filter.dim32(1) << " " << X.dim32(0) << " " << filter.dim32(2) << " " << filter.dim32(3) << std::endl;
      std::cout << ofm_conv0 << " " << src_shape.z() << " " << kernel_x_conv0 << " " << std::endl;
      if (filter.dim32(0) == ofm_conv0 && filter.dim32(1) == src_shape.z() && filter.dim32(2) == kernel_x_conv0 && filter.dim32(3) == kernel_x_conv0) {
      convertweight(weights, _weights, ofm_conv0, src_shape.z(), kernel_x_conv0);
      for (int i = 0; i < weights.size(); i++) {
         std::cout << unsigned(weights[i]) << " ";
      }
      std::cout << "start allocate weight data" << std::endl;
      weights0.allocator()->allocate(weights);
      std::cout << "finish initial weight data" << std::endl;
	}
      std::cout << "start get biases data" << std::endl;
      const uint8_t *_bias = bias.template data<uint8_t>();
      std::vector<uint8_t> biass(_bias, _bias + bias.size());
      biases0.allocator()->allocate(biass);
      std::cout << "end get biases data" << std::endl;

      out_conv0.allocator()->allocate();
      conv0.run();
      /*uint8_t *_res = weights0.allocator()->data();
      int len = weights0.allocator()->length();
      std::cout << "length of weight: " << len << std::endl;
      for (int i = 0; i < len; i++) {
	      std::cout << unsigned(*_res) << " ";
	      _res++;
      }*/
      uint8_t *_resa = out_conv0.allocator()->data();
      uint8_t *_result = Y->template mutable_data<uint8_t>();
      if (Y->dim32(0) == 1 && Y->dim32(1) == weight_shape[3]) {
	std::cout << "dim correct " << std::endl;
        convertres(_resa, _result, 1, weight_shape[3], oH, oW);
      }
    }
  private:
  };
  REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, ARM, ARMConvOp);
}
