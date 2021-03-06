/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.kubernetes.entrypoint;

import org.apache.flink.api.common.resources.CPUResource;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.TaskManagerOptions;
import org.apache.flink.kubernetes.configuration.KubernetesConfigOptions;
import org.apache.flink.util.TestLogger;

import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;

/** Tests for {@link KubernetesWorkerResourceSpecFactory}. */
public class KubernetesWorkerResourceSpecFactoryTest extends TestLogger {

    @Test
    public void testGetCpuCoresCommonOption() {
        final Configuration configuration = new Configuration();
        configuration.setDouble(TaskManagerOptions.CPU_CORES, 1.0);
        configuration.setDouble(KubernetesConfigOptions.TASK_MANAGER_CPU, 2.0);
        configuration.setDouble(KubernetesConfigOptions.TASK_MANAGER_CPU_LIMIT_FACTOR, 1.5);
        configuration.setInteger(TaskManagerOptions.NUM_TASK_SLOTS, 3);

        assertThat(
                KubernetesWorkerResourceSpecFactory.getDefaultCpus(configuration),
                is(new CPUResource(1.0)));
    }

    @Test
    public void testGetCpuCoresKubernetesOption() {
        final Configuration configuration = new Configuration();
        configuration.setDouble(KubernetesConfigOptions.TASK_MANAGER_CPU, 2.0);
        configuration.setDouble(KubernetesConfigOptions.TASK_MANAGER_CPU_LIMIT_FACTOR, 1.5);
        configuration.setInteger(TaskManagerOptions.NUM_TASK_SLOTS, 3);

        assertThat(
                KubernetesWorkerResourceSpecFactory.getDefaultCpus(configuration),
                is(new CPUResource(2.0)));
    }

    @Test
    public void testGetCpuCoresNumSlots() {
        final Configuration configuration = new Configuration();
        configuration.setInteger(TaskManagerOptions.NUM_TASK_SLOTS, 3);

        assertThat(
                KubernetesWorkerResourceSpecFactory.getDefaultCpus(configuration),
                is(new CPUResource(3.0)));
    }
}
