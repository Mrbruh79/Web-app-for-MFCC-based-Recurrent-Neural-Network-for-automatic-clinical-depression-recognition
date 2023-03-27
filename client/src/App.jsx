import "./App.css";
import { useState, useRef } from "react";
import AudioRecorder from "./components/AudioRecorder";
import Navigation from "./components/Navigation";
import HomePage from "./components/HomePage";

const App = () => {
      return (
        <div className=" flex flex-col items-stretch h-screen">
       {/* <Navigation />
      
       <HomePage /> */}
       
       <AudioRecorder/>
       </div>
       
       
    );
};
export default App;