import { useState, useRef,useEffect } from "react";
import axios from "axios";
import {v4 as uuidv4} from 'uuid';
import { Carousel } from "react-responsive-carousel";
import DocViewer from "react-doc-viewer";
import Slider from "react-slick";
import Slid from "./Slid";
import AudioUpload from "./audioUpload";
const AudioRecorder = () => {
    const myuuid = uuidv4();
    
const mediaRecorder = useRef(null);
const [recordingStatus, setRecordingStatus] = useState("inactive");
const [audioChunks, setAudioChunks] = useState([]);
const [audio, setAudio] = useState(null);
const mimeType = "audio/webm";
    const [permission, setPermission] = useState(false);
    const [stream, setStream] = useState(null);
    const [recorded,setRecorded]=useState(false);
const [generated,setGenerated]= useState(false);
const [data,setData]= useState({});
const sendUid = async ()=>{
    
        console.log("Sendiung the request");
        const formdata= new FormData();
        formdata.append('uid',myuuid);
       axios.post("http://localhost:5000/python",formdata).then((response)=>{
        console.log(response);
        setData(response.data);
        setGenerated(true);
       })
       console.log(data);
      
       
}
const docs = [
    { uri: "./Blank.pdf" } // Loc
  ];
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
        formData.append('uid',myuuid);
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
       
        <div style={{color:'#fff'}} className=" h-1/2 flex-col flex-wrap mx-auto space-y-16 font-Roboto text-lg items-stretch text-center mb-10">
            <h2 className=" font-bold text-3xl">Audio Recorder</h2>
            <main className="flex-col">
                <div className="my-3">
                    {!permission ? (
                        <button onClick={getMicrophonePermission} type="button" className="border-2 rounded-xl p-2 border-gray-50">
                            Get Microphone
                        </button>
                    ): null}
                    {permission && recordingStatus=="inactive" ? (
                        <button type="button" onClick={startRecording} className="border-2 rounded-xl p-2 border-gray-50">
                            Record
                        </button>
                    ): null}
                    {permission && recordingStatus=="recording" ?(
                        <button type="button" onClick={stopRecording} className="border-2 rounded-xl p-2 border-gray-50">
                            Stop
                        </button>
                    ):null}
                    
                </div>
                {audio? (<div className >
                        <audio src={audio} controls className="my-10"></audio>
                        <a download href={audio } className="border-2 rounded-xl p-2 border-gray-50 my-10">
                            Download Recording
                        </a>
                    </div>):"Audio Not Recorded Yet"}

                
            </main>
            <button onClick={sendUid }  className="border-2 rounded-xl p-2 border-gray-50 mx-auto mb-10">Generate</button>
            <AudioUpload uid={myuuid} />
            {generated?<div>
                <h3>You can possibly have depression. Please contact a medical professional as soon as possible</h3>
                <p> You have depression on a scale of {data.level} out of 24</p>
            </div>:null}
        </div>
        
    );
};
export default AudioRecorder;