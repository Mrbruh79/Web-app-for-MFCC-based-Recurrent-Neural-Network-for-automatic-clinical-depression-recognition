/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",


"./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
    fontFamily:{
      'Roboto': [ 'Roboto Mono', 'monospace']
    },
    gridTemplateColumns:{
      'main':'2fr 1fr'
    },
    gridTemplateRows:{
      'main':'2fr 1fr'
    },
    backgroundImage:{
      'imagesection':'url("https://source.unsplash.com/random/?funky/")'
    }
  },
  plugins: [],
}
