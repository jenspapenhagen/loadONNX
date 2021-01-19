package de.papenhagen;

import ai.onnxruntime.*;
import lombok.extern.log4j.Log4j;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@Log4j
@Path("/predition")
public class GreetingResource {

    private String loadModel(){
        //read ONNX model
        final OrtEnvironment environment = OrtEnvironment.getEnvironment();
        try {
            final String modelPath = "../model.onnx";
            /*
            * The run call expects a Map<String,OnnxTensor> where the keys match input node names
            * stored in the model. These can be viewed by calling session.getInputNames()
            * or session.getInputInfo() on an instantiated session.
            * The run call produces a Result object, which contains
            * a Map<String,OnnxValue> representing the output.
            * The Result object is AutoCloseable and can be used
            * in a try-with-resources statement to prevent references from leaking out.
            * Once the Result object is closed, all it's child OnnxValues are closed too.
            */
            final OrtSession session = environment.createSession(modelPath, new OrtSession.SessionOptions());
            if(session == null){
                log.error("model not found");
                return "";
            }
            final Map<String, NodeInfo> inputInfo = session.getInputInfo();
            //getShapeOfNodeInfo(inputInfo);
            //inputInfo.entrySet().forEach(System.out::println);
            //key: sequential_input:0 Value: TensorInfo(javaType=FLOAT, onnxType=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape=[-1, 469, 470, 3])

            final long[] shape = new long[] { -1L, 469L, 470L, 3L };
            //buffersize unknown have to get this info, need to check how
            float[] inputArr =
                    new float[] {
                            1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f, 13.0f,
                            -14.0f, 15
                    };
            FloatBuffer buffer = FloatBuffer.wrap(inputArr);

            OnnxTensor newTensor = OnnxTensor.createTensor(environment, buffer, shape);
            Map<String, OnnxTensor> input = new HashMap<>();
            input.put("0", newTensor);
            session.run(input);
        } catch (OrtException ex) {
            log.error("Error on creating the Session: " +ex.getMessage() );
        }

        return "test";
    }

    private void getShapeOfNodeInfo(Map<String, NodeInfo> inputInfo) {
        for (NodeInfo nodeInfo : inputInfo.values()) {
            if (nodeInfo.getInfo() instanceof TensorInfo){
                TensorInfo tempInfo = ((TensorInfo) nodeInfo.getInfo());
                Arrays.stream(tempInfo.getShape()).forEach(System.out::println);
            }else{
                log.info("not TensorInfo");
            }
        }
    }

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String predictions() {
        return "Hello RESTEasy" + loadModel();
    }
}