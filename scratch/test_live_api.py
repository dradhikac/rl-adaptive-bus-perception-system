import urllib.request
import json
import time

def test_health():
    res = urllib.request.urlopen("http://127.0.0.1:5000/api/health")
    data = json.loads(res.read())
    print("Health check:", data)

def test_image_analysis():
    # Test POST /api/analyze with red light sample
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open("data/red_light_sample.jpg", "rb") as f:
        img_bytes = f.read()
        
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="front"; filename="front.jpg"\r\n')
    body.extend(b'Content-Type: image/jpeg\r\n\r\n')
    body.extend(img_bytes)
    body.extend(b'\r\n')
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    
    t0 = time.time()
    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/analyze?model=stock&include_images=true",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    res = urllib.request.urlopen(req)
    t1 = time.time()
    resp_data = json.loads(res.read())
    print(f"Image analysis took: {t1 - t0:.2f}s")
    print(f"Decision: {resp_data.get('decision')}, Speed: {resp_data.get('recommended_speed')}, Signal: {resp_data.get('traffic_signal')}, Risk: {resp_data.get('collision_risk')}")
    assert resp_data.get("traffic_signal") == "RED", f"Expected RED signal, got {resp_data.get('traffic_signal')}"
    assert resp_data.get("decision") == "STOP", f"Expected STOP decision, got {resp_data.get('decision')}"
    assert resp_data.get("recommended_speed") == 0, f"Expected 0 speed, got {resp_data.get('recommended_speed')}"
    assert "annotated_front_image" in resp_data, "Expected annotated_front_image in response"
    print("Image test PASSED successfully!")

def test_video_analysis():
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open("tests/synthetic_test_clip.mp4", "rb") as f:
        vid_bytes = f.read()
        
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="front"; filename="front.mp4"\r\n')
    body.extend(b'Content-Type: video/mp4\r\n\r\n')
    body.extend(vid_bytes)
    body.extend(b'\r\n')
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="target_fps"\r\n\r\n')
    body.extend(b'1.0\r\n')
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    body.extend(b'stock\r\n')
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    
    t0 = time.time()
    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/demo/upload",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    res = urllib.request.urlopen(req)
    upload_res = json.loads(res.read())
    job_id = upload_res["job_id"]
    print("Video upload accepted, job_id:", job_id)
    
    # Poll status
    for i in range(30):
        time.sleep(1)
        st_res = urllib.request.urlopen(f"http://127.0.0.1:5000/api/demo/status/{job_id}")
        st_data = json.loads(st_res.read())
        status = st_data.get("status")
        progress = st_data.get("progress", 0)
        print(f"[{i+1}s] Job status: {status}, progress: {progress}% - {st_data.get('stage')}")
        if status == "completed":
            t_total = time.time() - t0
            print(f"Video analysis completed in {t_total:.2f}s!")
            print(f"Total timesteps: {st_data.get('total_timesteps')}")
            return
        elif status == "failed":
            raise RuntimeError(f"Video job failed: {st_data.get('error')}")
            
    raise TimeoutError("Video processing timed out")

if __name__ == "__main__":
    test_health()
    test_image_analysis()
    test_video_analysis()
