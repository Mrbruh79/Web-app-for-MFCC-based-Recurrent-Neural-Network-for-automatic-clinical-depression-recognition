import { useState, useRef,useEffect } from "react";
import axios from "axios";

const AudioRecorder = () => {
    
const mediaRecorder = useRef(null);
const [recordingStatus, setRecordingStatus] = useState("inactive");
const [audioChunks, setAudioChunks] = useState([]);
const [audio, setAudio] = useState(null);
const mimeType = "audio/webm";
    const [permission, setPermission] = useState(false);
    const [stream, setStream] = useState(null);
    const [recorded,setRecorded]=useState(false);

    const getMicrophonePermission = async () => {
        if ("MediaRecorder" in window) {
            try {
                const streamData = await navigator.mediaDevices.getUserMedia({
                    audio: true,
                    video: false,
                });
                setPermission(true);
                setStream(streamData);
            } catch (err) {
                alert(err.message);
            }
        } else {
            alert("The MediaRecorder API is not supported in your browser.");
        }
    };
    const startRecording= async () =>{
        setRecordingStatus("recording");
  //create new Media recorder instance using the stream
  const media = new MediaRecorder(stream, { type: mimeType });
  //set the MediaRecorder instance to the mediaRecorder ref
  mediaRecorder.current = media;
  //invokes the start method to start the recording process
  mediaRecorder.current.start();
  let localAudioChunks = [];
  mediaRecorder.current.ondataavailable = (event) => {
     if (typeof event.data === "undefined") return;
     if (event.data.size === 0) return;
     localAudioChunks.push(event.data);
     //console.log(localAudioChunks);
  };
  setAudioChunks(localAudioChunks);

    }
    const stopRecording =  ()=>{
        setRecordingStatus("inactive");
        mediaRecorder.current.stop();
        mediaRecorder.current.onstop = ()=>{
        const audioBlob = new Blob(audioChunks,{type:mimeType});
        const audioUrl= URL.createObjectURL(audioBlob);
        sendAudio(audioBlob)
        setRecorded(true);
        setAudio(audioUrl);
       console.log(audio);
        setAudioChunks([]);
        }
    }
    const sendAudio= async (audioFile)=>{
        console.log(audioFile);
        const audioBlob= audioFile;
        const formData = new FormData();
        console.log(audioFile.size);
   
        formData.append('audioFile', audioFile,"audio.weba");
        console.log(formData.get('audioFile'));
        const response = await axios.post('http://localhost:5000/audio', formData , {
            headers:{
                "Content-Type":"audio/webm"
            }
        })
         
      
        if (response.ok) {
          console.log(response);
        } else {
          console.error('Error uploading audio file.');
          console.log(response.body);
        }
  
    }
    
    
    return (
        <div style={{color:'#fff'}}>
            <h2>Audio Recorder</h2>
            <main>
                <div className="audio-controls">
                    {!permission ? (
                        <button onClick={getMicrophonePermission} type="button">
                            Get Microphone
                        </button>
                    ): null}
                    {permission && recordingStatus=="inactive" ? (
                        <button type="button" onClick={startRecording}>
                            Record
                        </button>
                    ): null}
                    {permission && recordingStatus=="recording" ?(
                        <button type="button" onClick={stopRecording}>
                            Stop
                        </button>
                    ):null}
                    
                </div>
                {audio? (<div className="audiocontainer" >
                        <audio src={audio} controls></audio>
                        <a download href={audio}>
                            Download Recording
                        </a>
                    </div>):"Audio Not Recorded Yet"}
            </main>
        </div>
    );
};
export default AudioRecorder;