MODULE InputDrawing
    VAR num x;
    VAR num y;
    VAR num z;
    VAR robtarget Object_Target;
    VAR pos p1;
    VAR num target;
    VAR string data;
    VAR socketdev client_socket;
    VAR socketdev temp_socket;
    VAR robtarget targetRobTarget;
    VAR string tempX;
    VAR string tempY;
    VAR string tempZ;
    VAR num idx0;
    VAR num idx1;
    VAR num idx2;
    VAR bool success;
    CONST robtarget home1:=[[409.328464947,30.699294352,-350.922061873],[0.999898286,-0.005230998,0.00469865,0.012408784],[0,-1,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget WorkSpaceCenter1:=[[9.78,391.21,-4.41],[0.988086,-0.00583922,0.00371725,-0.153745],[-1,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];

    PROC main()
        MoveJ home1,v1000,z100,tool2\WObj:=Wobj_1;
        SocketConnect;

        WHILE TRUE DO
            ! socket sent in "x,z,y"
            IF SocketGetStatus(client_socket)=SOCKET_CONNECTED THEN
                SocketReceive client_socket\Str:=data,\Time:=WAIT_MAX;
                SocketSend client_socket\Str:="R";
                ConvertSocketStrToPose(data);
                ! Wait until ConvertSocketStrToPose completes
                IF success THEN
                    MoveL Offs(WorkSpaceCenter1,x,y,z),v1000,z100,tool2\WObj:=Wobj_1;                
                    ! SocketSend client_socket\Str:="D";
                ENDIF
            ENDIF
        ENDWHILE
    ENDPROC

    ! for real life robot station
!    PROC SocketConnect()
!        SocketCreate temp_socket;
!        SocketBind temp_socket,"192.168.125.1",1025;
!        SocketListen temp_socket;
!        SocketAccept temp_socket,client_socket,\Time:=WAIT_MAX;
!        TPWrite "Socket connection established.";
!    ENDPROC

    ! for simulation station
    
    PROC SocketConnect()
        ! Create, bind, listen, and accept the socket connection
        SocketCreate temp_socket;
        SocketBind temp_socket,"127.0.0.1",55000;
        SocketListen temp_socket;
        SocketAccept temp_socket,client_socket,\Time:=WAIT_MAX;
        TPWrite "Socket connection established.";
    ENDPROC    
    PROC ConvertSocketStrToPose(string data)
        ! Find indices of the commas
        idx1:=StrFind(data,1,",");
        idx2:=StrFind(data,idx1+1,",");

        ! Ensure all indices are valid
        IF idx1>0 AND idx2>0 THEN
            ! Extract substrings for x, y, and z
            tempX:=StrPart(data,1,idx1-1);
            tempZ:=StrPart(data,idx1+1,idx2-idx1-1);
            tempY:=StrPart(data,idx2+1,StrLen(data)-idx2);
            ! Convert strings to numeric values
            success := FALSE;
            IF StrToVal(tempX,x) AND
                           StrToVal(tempZ,z) AND
                           StrToVal(tempY,y) THEN
                success := TRUE;
            ENDIF
        ELSE
            success := FALSE;
        ENDIF
    ENDPROC

ENDMODULE