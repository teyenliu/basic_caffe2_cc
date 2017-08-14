#include "caffe2/core/init.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {
  void run() {
    caffe2::Workspace ws;

    std::vector<float> x(4 * 3 * 2);
    for (auto &v : x) {
      v = (float)rand() / RAND_MAX;
    }

    {
      auto tensor = ws.CreateBlob("my_x")->GetMutable<TensorCPU>();
      auto value = TensorCPU({4, 3, 2}, x, NULL);
      tensor->ResizeLike(value);
      tensor->ShareData(value);
    }

    NetDef initModel;
    initModel.set_name("my_net");
    NetDef predictModel; 
    predictModel.set_name("my_pred");

    {
      auto op = predictModel.add_op();
      op->set_type("Conv");
      auto arg = op->add_arg();
      arg->set_name("kernel");
      arg->set_i(5);
      op->add_input("data");
      op->add_input("conv1_w");
      op->add_input("conv1_b");
      op->add_output("conv1");
    }


  }

}
  int main(int argc, char **argv) {
    caffe2::GlobalInit(&argc, &argv);
    caffe2::run();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
  }
