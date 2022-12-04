/*
 *  Copyright (c) 2021 enpasos GmbH
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package com.enpasos.bugs;


import ai.djl.Device;
import ai.djl.util.cuda.CudaUtils;
import lombok.Data;

import java.lang.management.MemoryUsage;

@Data
public class GPUMemoryChange {
    private long value;

    public synchronized void reset() {
        value = 0;
    }

    public synchronized void on() {
        value -= calculateValue();
    }

    public synchronized void off() {
        value += calculateValue();
    }


    private long calculateValue() {
        Device device = Device.gpu(0);
        MemoryUsage mem = CudaUtils.getGpuMemory(device);
        return mem.getCommitted();
    }

}
