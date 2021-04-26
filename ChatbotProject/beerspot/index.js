const config = require('../config');
const request = require('request');

const extractIntent = (nlp) => {
  
  if(nlp.intents[0]){

    if(nlp.intents[0].confidence > 0.8){
      return nlp.intents[0].name;
    }
    else{
      return null;
    }
  }
  else{
    return null;
  }
  
}

const extractEntity = (nlp , entity ) => {
  
  if(nlp.entities[entity]){

  
    if(nlp.entities[entity][0].confidence > 0.8){
      return nlp.entities[entity][0].value;
    }
    else{
      return null
    }
  }
  else{
    return null
  }
  

}
const getBeerData = (intent,beer_label = null , beer_type = null) => {
  // brewerydb apikey:9eacc3db222dd25489e232a482ef15b3

  return new Promise (( resolve , reject ) => {
    
    
    
    if(intent!=null){
      
      url = "https://sandbox-api.brewerydb.com/v2/beers?key=9eacc3db222dd25489e232a482ef15b3"  
      
      if(beer_label){
      url+="&name="+ beer_label
      }
      if(beer_type){
        if(beer_type == "strong"){
          url+="&abv=+7"
        }
        if(beer_type == "light"){
          url+="&abv=-7"
        }
        
      }

    }

    console.log(url)
    
    
    try{
      request(url, { json: true }, (err, res, body) => {
      if (err) {
        console.log(err)
        resolve(err) ; 
      }
      else{
        console.log( body)
        let title = body.data[0].name
        let alc = body.data[0].abv
        let description = body.data[0].style.description
        
        let img_path = "https://brewerydb-images.s3.amazonaws.com/beer/"+body.data[0].labels.medium

        infos = {title,alc,description,img_path}
        resolve(infos)
      }
      
      
      
      })
      
    }
    catch(error){
      reject(error)
    }
    
    


  }) ;
}

module.exports = nlpData => {
  
  return new Promise ( async function ( resolve , reject ) {
    let intent = extractIntent (nlpData , 'intent') ;
    console.log(intent + " intent");
    if( intent ) {

      let label = extractEntity ( nlpData , 'beer_label:beer_label') ;
      console.log(label + " label")
      let type = extractEntity ( nlpData , 'beer_type:beer_type') ;
      console.log(type + " type")
  // Get data ( including id) about the movie
      try {
        let beerData = await getBeerData (intent,label ,type ) ;
        resolve (beerData) ;
      } 
      catch ( error ) {
        resolve ( error ) ;
      }
 
   } 
    else {
    resolve ({
      txt: "I’m not sure I understand you!"
    }) ;
   }
  
   }) ;
}


