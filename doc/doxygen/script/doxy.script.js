
var backImg;
var con;

// === Copy text to clipboard ===
function cp2cb(text){
    copyTextToClipboard(text);
    con.log("\"" + text + "\" copied to clipboard.");

}

class erisConsole{
    constructor(){
        this.self = $("#console");
        this.content = $("#consoleContent");
        this.cnt = 0;
    }
    log(msg){
        this.self.stop();
        this.content.append("<div>[" + this.cnt + "] " + msg + "</div>");
        this.cnt++;
        this.self.animate({
            scrollTop: this.content.height()
        },0);

    }
}

class backgroundImage {
    constructor(){
        this.opacity_full = 0.3;
        this.opacity_partial = 0.1;
        this.opacity_low = 0.07;
        this.animation_speed = 200;
    }

    animate(){
        $("img#bottom_corner").stop(true);
        $("img#bottom_corner").animate({opacity: this.opacity_full},2000).animate({opacity: this.opacity_low},2000);
    }
    animateSearch(){
        $("img#bottom_corner").stop(true);
        $("img#bottom_corner").animate({opacity: this.opacity_full},this.animation_speed)
                .animate({opacity: this.opacity_partial},this.animation_speed)
                .animate({opacity: this.opacity_full},this.animation_speed)
                .animate({opacity: this.opacity_partial},this.animation_speed)
                .animate({opacity: this.opacity_full},this.animation_speed)
                .animate({opacity: this.opacity_partial},this.animation_speed)
                .animate({opacity: this.opacity_full},this.animation_speed)
                .animate({opacity: this.opacity_low},this.animation_speed);
    }
}

$(document).ready(function(){
    
    
    backImg = new backgroundImage();
    con = new erisConsole();
   
    function prepareSearchField(){
        if($("#MSearchField").length === 0){
            setTimeout(prepareSearchField, 1);
            console.log("Delay Search Field setup");
            return;
        }
        
        
        $("#MSearchField").focusin(function(){
            if( $("#MSearchField").val() === "Search ( press f )" || $("#MSearchField").val() === "Search" ) {
                $("#MSearchField").val("");
            }
        });
        $("#MSearchField").blur(function(){
            if( $("#MSearchField").val() === "" || $("#MSearchField").val() === "Search" ) {
                $("#MSearchField").val("Search ( press f )");
            }
        });
        $("#MSearchField").focusout(function(){
            if( $("#MSearchField").val() === "" || $("#MSearchField").val() === "Search" ) {
                $("#MSearchField").val("Search ( press f )");
            }
        });
        $("#MSearchField").change(function(){
            if( $("#MSearchField").val() === "" || $("#MSearchField").val() === "Search" ) {
                $("#MSearchField").val("Search ( press f )");
            }
        });
        
        
        $("#MSearchField").val("Search ( press f )");
        
    }
    
    var arrow_length = 0;
    var arrow_counter = 0;
    
    function prepareNavigation(){
//        if( $("span.arrow").length === 0 ){
//            setTimeout(prepareNavigation, 5);
//            console.log("Delay Navigation setup");
//            return;
//        }
//        
//        if( $("span.arrow").length === arrow_length && arrow_counter < 100){
//            arrow_counter++;
//            setTimeout(prepareNavigation, 5);
//            console.log("Delay Navigation setup");
//            return;
//        }
//        
//        arrow_counter = 0;
//        $("span.arrow").html("");
        
        
        
//        if($("div#nav-tree").css("display") === "none"){
//            setTimeout(prepareNavigation, 1);
//            console.log("Delay Navigation setup");
//            return;
//        }
//        console.log($("span.arrow"));
    }
    
    prepareSearchField();
//    prepareNavigation();
    $("div#blur").remove();
    $("div#noJSWarning").remove();
    
    $("div#nav-tree").css("display", "block");
    
//    $("span.arrow").html("");
    
    
    
//    setTimeout(function(){
//        $("#MSearchField").focusin(function(){
//            if( $("#MSearchField").val() === "Search ( press f )" || $("#MSearchField").val() === "Search" ) {
//                $("#MSearchField").val("");
//            }
//        });
//        $("#MSearchField").blur(function(){
//            if( $("#MSearchField").val() === "" || $("#MSearchField").val() === "Search" ) {
//                $("#MSearchField").val("Search ( press f )");
//            }
//        });
//        $("#MSearchField").val("Search ( press f )");
//        
//    }, 500);
    
    
    /*
     * Expand container functionality
     * 
     * Enables to un-/fold container created using the command @expandable
     */
    $("div.expandable").each(function(index){
        $(this).children("div.exp_content").slideUp(0);
        var trigger = $(this).children("span.exp_trigger");
        var arrow = $(this).children("span.exp_arrow");
        var content = $(this).children("div.exp_content");
        var func = function(){
           if(content.css("display") === "block"){
               arrow.html("&rArr;");
               trigger.text("Click to expand");
           } else {
               arrow.html("&dArr;");
               trigger.text("Click to hide");
           }
           content.slideToggle(800); 
        };

        trigger.click(func);
        arrow.click(func);
        
    });
    
    $(document).keypress(function(event){
        // f
        if(event.which === 102) {
            // select search bar
            if( ! $("#MSearchField").is(":focus") ){
                $("#MSearchField").select();
            }
            
        }
        // e
        if(event.which === 101) {
            backImg.animate();
        }
        // g
        if(event.which === 103) {
            console.log($("#MSearchField"));
        }
        
        if($("#MSearchField").is(":focus") && event.which !== 8 && $("#MSearchField").val() !== ""){
            backImg.animateSearch();
        }
        
        console.log("[DEV] key pressed: " + event.which);
    });
    

    
    
    var pulseInterval = setInterval(function(){
        backImg.animate();
    },64000);
    
    $("#MSearchField").unbind("blur");
    $("#MSearchField").focusout(function(){
       console.log("focus out"); 
    });
    $("#MSearchField").blur(function(){
       console.log("blur out"); 
    });
    
    if( $("#doc-content").height() < parseInt($("div.contents").css("margin-top"), 10) + $("div.contents").height() + $("div.header").height() - 1 ){
        // todo
        $("img#bottom_corner").css("right", "15px");
        console.log($("#doc-content").height() ," zu ", parseInt($("div.contents").css("margin-top"), 10) + $("div.contents").height() + $("div.header").height() - 1);
    }
    
    $("span#Fridolin").click(function(){
        var div = $("div#Fridolin");
        var frido = div.children("img:first-child");
        var bubble = div.children("img:nth-child(2)");
        
        div.css("display","block").animate({
            bottom: "-55px"
        }, 500, function(){
            bubble.css("display","block").animate({
                opacity: 1
            },200, function(){
                bubble.delay(200).animate({
                    opacity: 0
                },100, function(){
                    bubble.css("display","none");
                    div.animate({
                        bottom: "-235px"
                    },500, function(){
                        div.css("display","none");
                    });
                });
            });
        });
        
    });
    
    if( $("div.header").length !== 0 ) {
        $("div.contents").css("min-height","calc(100% - " + ( parseInt($("div.header").height(),10) + 1 + 20 ) + "px)");
    } 
    
//    Cookies.set('test', "abcdefg");
    
//    console.log(Cookies.get("test"));

    
//    $(".copy").click(function(){
//        copyTextToClipboard($(this).text());
//        con.log("Copied to clipboard.");
//   });
    
    
    
}); // document ready
