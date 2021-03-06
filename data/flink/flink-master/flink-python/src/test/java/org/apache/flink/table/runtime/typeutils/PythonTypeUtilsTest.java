/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.table.runtime.typeutils;

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.fnexecution.v1.FlinkFnApi;
import org.apache.flink.table.catalog.UnresolvedIdentifier;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.UnresolvedUserDefinedType;
import org.apache.flink.util.ExceptionUtils;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Tests for {@link PythonTypeUtils}. */
public class PythonTypeUtilsTest {

    @Test
    public void testLogicalTypetoInternalSerializer() {
        List<RowType.RowField> rowFields = new ArrayList<>();
        rowFields.add(new RowType.RowField("f1", new BigIntType()));
        RowType rowType = new RowType(rowFields);
        TypeSerializer baseSerializer = PythonTypeUtils.toInternalSerializer(rowType);
        assertTrue(baseSerializer instanceof RowDataSerializer);

        assertEquals(1, ((RowDataSerializer) baseSerializer).getArity());
    }

    @Test
    public void testLogicalTypeToProto() {
        List<RowType.RowField> rowFields = new ArrayList<>();
        rowFields.add(new RowType.RowField("f1", new BigIntType()));
        RowType rowType = new RowType(rowFields);
        FlinkFnApi.Schema.FieldType protoType =
                rowType.accept(new PythonTypeUtils.LogicalTypeToProtoTypeConverter());
        FlinkFnApi.Schema schema = protoType.getRowSchema();
        assertEquals(1, schema.getFieldsCount());
        assertEquals("f1", schema.getFields(0).getName());
        assertEquals(
                FlinkFnApi.Schema.TypeName.BIGINT, schema.getFields(0).getType().getTypeName());
    }

    @Test
    public void testLogicalTypeToDataConverter() {
        PythonTypeUtils.DataConverter converter = PythonTypeUtils.toDataConverter(new IntType());

        GenericRowData data = new GenericRowData(1);
        data.setField(0, 10);
        Object externalData = converter.toExternal(data, 0);
        assertTrue(externalData instanceof Long);
        assertEquals(externalData, 10L);
    }

    @Test
    public void testUnsupportedTypeSerializer() {
        LogicalType logicalType =
                new UnresolvedUserDefinedType(UnresolvedIdentifier.of("cat", "db", "MyType"));
        String expectedTestException =
                "Python UDF doesn't support logical type `cat`.`db`.`MyType` currently.";
        try {
            PythonTypeUtils.toInternalSerializer(logicalType);
        } catch (Exception e) {
            assertTrue(
                    ExceptionUtils.findThrowableWithMessage(e, expectedTestException).isPresent());
        }
    }
}
