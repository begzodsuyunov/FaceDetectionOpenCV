package com.example.opencv_facedetection;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final int MY_PERMISSIONS_REQUEST_CAMERA = 1001;
    File casFile;
    JavaCameraView javaCameraView;
    CascadeClassifier faceDetector;
    private Mat mRgba, mGray;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        javaCameraView = findViewById(R.id.javaCamView);

        // Check camera permission at runtime
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},
                    MY_PERMISSIONS_REQUEST_CAMERA);
        } else {
            // Permission already granted, initialize OpenCV
            initializeOpenCV();
        }

        // Initialize JavaCameraView
        javaCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);

        javaCameraView.setCvCameraViewListener(this);


    }
    private void initializeOpenCV() {
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseCallback);
            Toast.makeText(this, "OpenCV initialization in progress", Toast.LENGTH_SHORT).show();
        } else {
            baseCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == MY_PERMISSIONS_REQUEST_CAMERA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Camera permission granted, initialize OpenCV
                initializeOpenCV();
            } else {
                // Camera permission denied, show a message or handle accordingly
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Nullable
    @Override
    public Object onRetainCustomNonConfigurationInstance() {
        return faceDetector;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        mRgba = new Mat();
        mGray = new Mat();


    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }
    private int frameCount = 0;

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        if (frameCount % 2 == 0) {

            mRgba = inputFrame.rgba();
            mGray = inputFrame.gray();

            // Ensure faceDetector is initialized
            if (faceDetector != null) {
                MatOfRect faceDetections = new MatOfRect();
                faceDetector.detectMultiScale(mGray, faceDetections, 1.5, 4, 0, new Size(100, 100), new Size());
//                faceDetector.detectMultiScale(mGray, faceDetections, 1.9, 4, 0, new Size(100, 100), new Size());
                /**
                 * What is scale factor?
                 * The scale factor is crucial for performance.
                 * A smaller scale factor can increase the number of image scales to search, potentially improving detection accuracy but at the cost of increased computation.
                 * Which means that lower the scale factor the search is more accurate but it results in lag
                 */

//                faceDetector.detectMultiScale(mRgba, faceDetections);


                for (Rect rect : faceDetections.toArray()) {
                    Imgproc.rectangle(mRgba, new Point(rect.x, rect.y),
                            new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(255, 0, 0), 4);
                }
            }
        }
        frameCount++;

        return mRgba;
    }


    private BaseLoaderCallback baseCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {

            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.d("MainActivity", "OpenCV initialization succeeded");

                    InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    casFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
//                    casFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");

                    FileOutputStream fos = null;
                    try {
                        fos = new FileOutputStream(casFile);
                    } catch (FileNotFoundException e) {
                        throw new RuntimeException(e);
                    }

                    byte[] buffer = new byte[4096];
                    int bytesRead;

                    try {
                        while ((bytesRead = is.read(buffer)) != -1) {
                            fos.write(buffer, 0, bytesRead);
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    try {
                        is.close();
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    try {
                        fos.close();
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }

                    faceDetector = new CascadeClassifier(casFile.getAbsolutePath());

                    if (faceDetector.empty()) {
                        faceDetector = null;
                    } else {
                        cascadeDir.delete();
                    }

                    runOnUiThread(() -> {
                        if (javaCameraView != null) {
                            javaCameraView.enableView();
                            Log.d("enablec", "started");
                        }
                    });

                }
                break;
                default: {
                    Log.e("MainActivity", "OpenCV initialization failed with status: " + status);

                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }
}