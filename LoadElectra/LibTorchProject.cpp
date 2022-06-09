#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

//int main(int argc, const char* argv[]) {
//    if (argc != 2) {
//        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
//        return -1;
//    }
//
//
//    torch::jit::script::Module module;
//    try {
//        // Deserialize the ScriptModule from a file using torch::jit::load().
//        module = torch::jit::load(argv[1]);
//    }
//    catch (const c10::Error& e) {
//        std::cerr << "error loading the model\n";
//        return -1;
//    }
//
//    std::cout << "ok\n";
//}
int main() {
    //if (argc != 2) {
    //    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    //    return -1;
    //}


    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("C:\\Users\\ejpark\\Desktop\\vs\\LoadElectra\\traced_electra.pt"); // 경로 절대경로로 박기.. 

    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }


    // 입력값 벡터를 생성합니다.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({ 1, 3, 224, 224 }));

    // 모델을 실행한 뒤 리턴값을 텐서로 변환합니다.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    std::cout << "ok\n";
}