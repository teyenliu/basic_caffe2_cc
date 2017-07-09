#include "caffe2/core/init.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/net.h"


namespace caffe2 {

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 Intro Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/intro-tutorial.html" << std::endl;
  std::cout << std::endl;

  // >>> from caffe2.python import workspace, model_helper
  // >>> import numpy as np
  Workspace workspace;

  // >>> x = np.random.rand(4, 3, 2)
  std::vector<float> x(4 * 3 * 2);
  for (auto &v: x) {
    v = (float)rand() / RAND_MAX;
  }


  // >>> workspace.FeedBlob("my_x", x)
  {
    auto tensor = workspace.CreateBlob("my_x")->GetMutable<TensorCPU>();
    auto value = TensorCPU({ 4, 3, 2 }, x, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> x2 = workspace.FetchBlob("my_x")
  // >>> print(x2)
  {
    const auto blob = workspace.GetBlob("my_x");
    auto tensor = blob->Get<Tensor<CPUContext>>();
    const auto& data = tensor.template data<float>();
    std::cout << "my_net" << "(" << tensor.dims() << "): ";
    for (auto i = 0; i < (tensor.size() > 100 ? 100 : tensor.size()); ++i) {
        std::cout << (float)data[i] << ' ';
    }
    std::cout << std::endl;
  }

}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
