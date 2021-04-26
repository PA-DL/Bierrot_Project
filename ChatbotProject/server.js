'use strict';
require('dotenv').config();
const config = require('./config');
const FBeamer = require('./fbeamer');
const nlpData = require('./beerspot')
const express = require('express');
const bodyparser=require("body-parser");


const f = new FBeamer (config.FB) ;
const server = express();
server.use(bodyparser.json())
const PORT = process.env.PORT || 3000;


server.get ('/', (req , res ) => f.registerHook (req , res )) ;
server.listen(PORT, () => console.log(`The bot server is runnning on port ${PORT}`));
server.post('/',(req,res,next) => { 
  

  

  return f.incoming(req,res,async data => {
    
    const nlp = await data.content.nlp
    console.log(nlp)
    console.log(nlp.entities)
    let beer_infos = await nlpData(nlp)
    console.log(beer_infos)


    try {

        if(nlp.intents[0].name == "beer_alc_rate") {
            if(beer_infos.errno ){
              await f.txt(data.sender, "Sorry Brewery DB API is currently unavailable... We can't get the alcool amount in your beer. 😕");
              await f.sendImage(data.sender, "https://ctl.s6img.com/society6/img/oNDDqke9K4f19vLYGDnTDdrXk8U/w_700/prints/~artwork/s6-original-art-uploads/society6/uploads/misc/e035b39bdf1f4c8abd5884384c4f74b4/~~/error-404-beer-not-found-prints.jpg");
              await f.txt(data.sender, "intent: " + nlp.intents[0].name);
            }
            else{
              console.log("Alcool Rate");
              await f.txt(data.sender, beer_infos.alc);
              await f.sendImage(data.sender, beer_infos.img_path);
              await f.txt(data.sender, "If you need any other informations don't hesitate to ask the bot 😎 ");
              await f.txt(data.sender, "intent: " + nlp.intents[0].name);
            }
            
            
        }
        if(nlp.intents[0].name == "beer_recomendation") {
            if(beer_infos.errno ){
              await f.txt(data.sender, "Sorry, our Chatbot isn't link to our recommendation algorithm  😕");
              await f.txt(data.sender, "intent: " + nlp.intents[0].name);
            }
            else{
              console.log("Recommendation");
              await f.txt(data.sender, beer_infos.alc);
              await f.sendImage(data.sender, beer_infos.img_path);
              await f.txt(data.sender, "If you need any other informations don't hesitate to ask the bot 😎 ");
              await f.txt(data.sender, "intent: " + nlp.intents[0].name);
            }
            
            
        }
        if(nlp.intents[0].name == "beer_description") {
            if(beer_infos.errno ){
              await f.txt(data.sender, "Sorry Brewery DB API is currently unavailable... We can't get the description of your beer. 😕");
              await f.sendImage(data.sender, "https://ctl.s6img.com/society6/img/oNDDqke9K4f19vLYGDnTDdrXk8U/w_700/prints/~artwork/s6-original-art-uploads/society6/uploads/misc/e035b39bdf1f4c8abd5884384c4f74b4/~~/error-404-beer-not-found-prints.jpg");
              await f.txt(data.sender, "intent: " + nlp.intents[0].name);
            }
            else{
              console.log("Dexscription: ");
              await f.txt(data.sender, beer_infos.description);
              await f.sendImage(data.sender, beer_infos.img_path);
              await f.txt(data.sender, "If you need any other informations don't hesitate to ask the bot 😎 ");
              await f.txt(data.sender, "intent: " + nlp.intents[0].name);
            }
            
            
        }
        else if (nlp.intents[0].name == "hello"){
          await f.txt(data.sender, "Welcome to Bierrot. You can find here ideas about beers and get the description of your favorite beer 🍺😋  \n You can ask information about a specific beer \n Get the alcool rate of a beer \n Get the consumation occasions for this beer \n Get beer recomendations");
          await f.sendImage(data.sender, "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/fakebeers-1563897135.jpg?crop=1.00xw:1.00xh;0,0&resize=980:*");
          await f.txt(data.sender, "intent: " + nlp.intents[0].name);
        }

        else if (nlp.intents[0].name == "beer_consumation_occasions") {
          if(beer_infos.errno ){
              await f.txt(data.sender, "Sorry Brewery DB API is currently unavailable... We can't tell you the occasions to drink your beer. 😕 ");
              await f.sendImage(data.sender, "https://ctl.s6img.com/society6/img/oNDDqke9K4f19vLYGDnTDdrXk8U/w_700/prints/~artwork/s6-original-art-uploads/society6/uploads/misc/e035b39bdf1f4c8abd5884384c4f74b4/~~/error-404-beer-not-found-prints.jpg");
              await f.txt(data.sender, "intent: " + nlp.intents[0].name);
            }
          else{
            console.log("Drink occasion");
            if(int(beer_infos.alc) > 7){
              await f.txt(data.sender, "Prefer special occasions to drink "+beer_infos.title+" 🥳🥴");
              await f.sendImage(data.sender, beer_infos.img_path);
              await f.txt(data.sender, "If you need any other informations don't hesitate to ask the bot 😎 ");
              await f.txt(data.sender, "intent: " + nlp.intents[0].name);
            }
            else{
              await f.txt(data.sender, "No special occasion to drink "+beer_infos.title+" 😇");
              await f.sendImage(data.sender, beer_infos.img_path);
              await f.txt(data.sender, "If you need any other informations don't hesitate to ask the bot 😎 ");
              await f.txt(data.sender, "intent: " + nlp.intents[0].name);
            }
            
          }
          
        }
        
    } catch (e) {
        console.log(e);
        await f.txt(data.sender,"If you want informations about a beer enter a valid beer name or ask for recommendation ! 😊")
    }
    
    
    })

});
