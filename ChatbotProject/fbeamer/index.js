'use strict'
const request = require('request');
const apiVersion = 'v6.0';
class FBeamer{

    constructor({pageAccessToken, VerifyToken}){
        
        this.pageAccessToken = pageAccessToken;
        this.VerifyToken = VerifyToken;
    }

    messageHandler (obj){
        let sender = obj.sender.id;
        let message = obj.message;
        if(message.text){
            let obj = {
                sender,
                type:'text',
                content : message
            }
        return obj;
        }
    }

    registerHook(req, res) {
        const params = req.query;
        const mode = params['hub.mode'];
        const token = params['hub.verify_token'];
        const challenge = params['hub.challenge'];
       
        try{
            if((mode == 'subscribe') && (token == this.VerifyToken)){
                
                console.log('The webhook is registered');
                return res.send(challenge);
            } else {
                console.log("Could not register webhook!");
                return res.sendStatus(200);
            }
        } catch(e) {
            console.log(e);
        }
    }
    incoming(req,res,cb){
        res.sendStatus(200);
        if(req.body.object == 'page' && req.body.entry){
            let text = req.body.entry[0].messaging[0].message.text;
            let data = req.body
            
            data.entry.forEach(pageObj => {
                if(pageObj.messaging){
                    pageObj.messaging.forEach(messageObj=>{
                        if(messageObj.postback){

                        }
                        else{
                            return cb(this.messageHandler(messageObj))
                        }
                    })
                }
            });
            
            // to be continued
        }
    }

    sendMessage(payload) {
        return new Promise((resolve, reject) => {
            request({
                url : `https://graph.facebook.com/${apiVersion}/me/messages`,
                qs : {
                    access_token : this.pageAccessToken
                },
                method : 'POST',
                json : payload
            }, (error, response, body) => {
                if(!error && response.statusCode == 200) {
                    resolve({
                        mid : body.message_id
                    });
                } else {
                    reject(error);
                }
            });
        });
    }

    txt(id, text, messaging_type = 'RESPONSE') {
        let obj = {
            messaging_type,
            recipient : {
                id
            },
            message : {
                text
            }
        }

        return this.sendMessage(obj);
    }

    //Generic function sending images
    sendImage(id, url, messaging_type = 'RESPONSE') {
      
      let message = {
          "attachment": {
              "type": "image",
              "payload": {
                  "url": url
              } 
          }
      };


      let obj = {
            messaging_type,
            recipient : {
                id
            },
            message
            };
      return this.sendMessage(obj);
    };
}



module.exports = FBeamer;