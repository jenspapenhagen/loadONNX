package de.papenhagen;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import lombok.extern.slf4j.Slf4j;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Map;

@Slf4j
@Path("/prediction")
public class PredictionResource {

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String prediction() {
        return "Hello RESTEasy: " + loadModel();
    }

    private String loadModel() {
        //read ONNX model
        final OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();
        try {
            final String modelPath = "../model.onnx";
            /*
             * The run call expects a Map<String,OnnxTensor> where the keys match input node names
             * stored in the model. These can be viewed by calling environmentSession.getInputNames()
             * or environmentSession.getInputInfo() on an instantiated environmentSession.
             * The run call produces a Result object, which contains
             * a Map<String,OnnxValue> representing the output.
             * The Result object is AutoCloseable and can be used
             * in a try-with-resources statement to prevent references from leaking out.
             * Once the Result object is closed, all it's child OnnxValues are closed too.
             */
            final OrtSession environmentSession = ortEnvironment.createSession(modelPath, new OrtSession.SessionOptions());
            if (environmentSession == null) {
                log.error("model not found: " + modelPath);
                return "";
            }
            //for debugging
            //final Map<String, NodeInfo> inputInfo = environmentSession.getInputInfo();
            //final Map<String, NodeInfo> outputInfo = environmentSession.getOutputInfo();
            //getShapeOfNodeInfo(inputInfo);
            //key: sequential_input:0 Value: TensorInfo(javaType=FLOAT,
            // onnxType=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape=[-1, 469, 470, 3])

            //buffer-size unknown. have to get this info
            float[] inputArray =
                    new float[]{
                            1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f, 13.0f,
                            -14.0f, 15
                    };
            final FloatBuffer buffer = FloatBuffer.wrap(inputArray);

            final long[] inputShape = new long[]{-1L, 469L, 470L, 3L};

            final OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, buffer, inputShape);
            final Map<String, OnnxTensor> input = Map.of("0", onnxTensor);

            final OrtSession.Result result = environmentSession.run(input);
            final OnnxValue onnxValue = result.get(0);

            return Arrays.deepToString((float[][]) onnxValue.getValue());

        } catch (OrtException ex) {
            log.error("Error on creating the Session: " + ex.getMessage());
        }

        return "test";
    }

    /*
    * only for debugging
   */
    private void getShapeOfNodeInfo(Map<String, NodeInfo> infoMap) {
        for (NodeInfo nodeInfo : infoMap.values()) {
            if (nodeInfo.getInfo() instanceof TensorInfo tempInfo) {
                Arrays.stream(tempInfo.getShape()).forEach(System.out::println);
            } else {
                log.info("not TensorInfo");
            }
        }
    }

}