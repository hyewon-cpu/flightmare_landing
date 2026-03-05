#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

#include <flightlib/bridges/unity_bridge.hpp>
#include <flightlib/common/quad_state.hpp>
#include <flightlib/objects/quadrotor.hpp>

using namespace flightlib;

static void printState(const QuadState& state, const int step) {
  std::cout << "[state] step=" << step
            << " p=(" << state.p.x() << ", " << state.p.y() << ", " << state.p.z() << ")"
            << " v=(" << state.v.x() << ", " << state.v.y() << ", " << state.v.z() << ")"
            << " w=(" << state.w.x() << ", " << state.w.y() << ", " << state.w.z() << ")"
            << std::endl;
}

int main() {
  std::cout << "[cpp_control] starting..." << std::endl;

  // UnityBridge는 내부적으로 ZMQ로 Unity와 통신
  auto bridge = UnityBridge::getInstance();

  // Quadrotor 생성
  std::shared_ptr<Quadrotor> quad = std::make_shared<Quadrotor>();

  // 초기 상태
  QuadState state;
  state.setZero();
  state.p << 0, 0, 20;   // x,y,z (z=5m) 
  // -> unity (x,y,z)에서 y z 바꿔줘야함. unity에선 y가 위아래방향
  quad->reset(state);

  // Unity에 등록
  bridge->addQuadrotor(quad);

  // Unity와 핸드셰이크 + scene 명시 연결
  // WAREHOUSE = 1 (임시)
  // INDUSTRIAL = 0
  const SceneID scene_id = UnityScene::WAREHOUSE;
  if (!bridge->connectUnity(scene_id)) {
    std::cerr << "[cpp_control] failed to connect unity." << std::endl;
    return 1;
  }

  std::cout << "[cpp_control] quad added. rendering loop..." << std::endl;

  // 간단 루프: 조금씩 상승시키면서 렌더 요청
  for (int i = 0; i < 10000; i++) {
    quad->getState(&state);
    printState(state, i);
    state.p.z() += 0.05;     // 천천히 상승
    quad->setState(state);

    // Unity 프레임 요청 (true: 블로킹 렌더)
    bridge->getRender(true);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  return 0;
}
