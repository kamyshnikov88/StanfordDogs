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

package org.apache.flink.runtime.rpc.akka;

import org.apache.flink.api.common.time.Time;
import org.apache.flink.runtime.concurrent.akka.AkkaFutureUtils;
import org.apache.flink.runtime.rpc.RpcEndpoint;
import org.apache.flink.runtime.rpc.RpcGateway;
import org.apache.flink.runtime.rpc.RpcService;
import org.apache.flink.runtime.rpc.RpcTimeout;
import org.apache.flink.util.IOUtils;
import org.apache.flink.util.concurrent.FutureUtils;

import akka.actor.ActorSystem;
import akka.actor.Terminated;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Tests that ask timeouts report the call stack of the calling function. */
class TimeoutCallStackTest {

    private static ActorSystem actorSystem;
    private static RpcService rpcService;

    private final List<RpcEndpoint> endpointsToStop = new ArrayList<>();

    @BeforeAll
    static void setup() {
        actorSystem = AkkaUtils.createDefaultActorSystem();
        rpcService =
                new AkkaRpcService(actorSystem, AkkaRpcServiceConfiguration.defaultConfiguration());
    }

    @AfterAll
    static void teardown() throws Exception {

        final CompletableFuture<Void> rpcTerminationFuture = rpcService.stopService();
        final CompletableFuture<Terminated> actorSystemTerminationFuture =
                AkkaFutureUtils.toJava(actorSystem.terminate());

        FutureUtils.waitForAll(Arrays.asList(rpcTerminationFuture, actorSystemTerminationFuture))
                .get(10_000, TimeUnit.MILLISECONDS);
    }

    @AfterEach
    void stopTestEndpoints() {
        endpointsToStop.forEach(IOUtils::closeQuietly);
    }

    @Test
    void testTimeoutException() throws Exception {
        final TestingGateway gateway = createTestingGateway();

        final CompletableFuture<Void> future = gateway.callThatTimesOut(Time.milliseconds(1));

        assertThatThrownBy(future::get)
                .hasCauseInstanceOf(TimeoutException.class)
                .hasStackTraceContaining("testTimeoutException")
                .extracting(Throwable::getCause)
                .extracting(Throwable::getMessage)
                .satisfies(s -> assertThat(s).contains("callThatTimesOut"));
    }

    // ------------------------------------------------------------------------
    //  setup helpers
    // ------------------------------------------------------------------------

    private TestingGateway createTestingGateway() throws Exception {
        final TestingRpcEndpoint endpoint = new TestingRpcEndpoint(rpcService, "test_name");
        endpointsToStop.add(endpoint);
        endpoint.start();

        return rpcService.connect(endpoint.getAddress(), TestingGateway.class).get();
    }

    // ------------------------------------------------------------------------
    //  testing mocks / stubs
    // ------------------------------------------------------------------------

    private interface TestingGateway extends RpcGateway {

        CompletableFuture<Void> callThatTimesOut(@RpcTimeout Time timeout);
    }

    private static final class TestingRpcEndpoint extends RpcEndpoint implements TestingGateway {

        TestingRpcEndpoint(RpcService rpcService, String endpointId) {
            super(rpcService, endpointId);
        }

        @Override
        public CompletableFuture<Void> callThatTimesOut(@RpcTimeout Time timeout) {
            // return a future that never completes, so the call is guaranteed to time out
            return new CompletableFuture<>();
        }
    }
}
