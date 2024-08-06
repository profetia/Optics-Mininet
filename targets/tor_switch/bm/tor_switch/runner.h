/* Copyright 2018-present Barefoot Networks, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Antonin Bas (antonin@barefootnetworks.com)
 *
 */

#ifndef TOR_SWITCH_BM_TOR_SWITCH_RUNNER_H_
#define TOR_SWITCH_BM_TOR_SWITCH_RUNNER_H_

#include <bm/bm_sim/dev_mgr.h>
#include <bm/bm_sim/device_id.h>
#include <bm/bm_sim/options_parse.h>

#include <cstdint>
#include <memory>

class TorSwitch;

namespace bm {

namespace sswitch {

class TorSwitchRunner {
 public:
  static constexpr uint32_t default_drop_port = 511;

  explicit TorSwitchRunner(uint32_t cpu_port = 0,
                              uint32_t drop_port = default_drop_port);
  ~TorSwitchRunner();

  int init_and_start(const bm::OptionsParser &parser);

  device_id_t get_device_id() const;

  DevMgr *get_dev_mgr();

 private:
  uint32_t cpu_port{0};
  std::unique_ptr<TorSwitch> tor_switch;
};

}  // namespace sswitch

}  // namespace bm

#endif  // TOR_SWITCH_BM_TOR_SWITCH_RUNNER_H_
