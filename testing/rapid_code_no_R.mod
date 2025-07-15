MODULE AUTO_InputDrawing
    ! Socket and other declarations
    VAR socketdev client_socket;
    VAR socketdev temp_socket;
    
    ! Constant Targets
    CONST robtarget home1:=[[409.328464947,30.699294352,-350.922061873],[0.999898286,-0.005230998,0.00469865,0.012408784],[0,-1,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget WorkSpaceCenter1:=[[75.78,312.76,9.799641871],[0.988089954,-0.00592235,0.00373461,-0.153717993],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Pre_pack := [[121.48,454.11,-217.7],[0.750819,-0.242807,0.193622,0.582946],[0,-1,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    TASK PERS wobjdata Wobj_1:=[FALSE,TRUE,"",[[87.974520519,-126.434467699,0],[0,0.707106781,0.707106781,0]],[[0,0,0],[1,0,0,0]]];

    ! Main robot routine
    PROC main()
        ! Local variables for the coordinates
        VAR num x_val;
        VAR num y_val;
        VAR num z_val;
        VAR rawbytes received_bytes;
        VAR num read_pos;

        MoveJ home1,v1000,z100,tool2\WObj:=Wobj_1;
        SocketConnect;

        WHILE TRUE DO
            ! Reset read position for each new message
            read_pos := 1;
            
            ! Receive exactly 12 raw bytes from the client
            SocketReceive client_socket\RawData:=received_bytes, \ReadNoOfBytes:=12, \Time:=WAIT_MAX;
            
            ! ! Send "Ready" confirmation (1 byte)
            ! SocketSend client_socket\Str:="R" \NoOfBytes:=1;
            
            ! Unpack X from the starting position
            UnpackRawBytes received_bytes\Network, read_pos, x_val\Float4;
            ! Manually advance the read position by 4 bytes (the size of a Float4)
            read_pos := read_pos + 4;

            ! Unpack Y from the new position
            UnpackRawBytes received_bytes\Network, read_pos, y_val\Float4;
            ! Manually advance the read position by 4 bytes
            read_pos := read_pos + 4;

            ! Unpack Z from the final position
            UnpackRawBytes received_bytes\Network, read_pos, z_val\Float4;

            ! Optional: You can keep these for debugging on the Teach Pendant
            TPWrite "Received X:" \Num:= x_val;
            TPWrite "Received Y:" \Num:= y_val;
            TPWrite "Received Z:" \Num:= z_val;

            ! Perform the move using the standardized (X, Y, Z) offsets
            MoveL Offs(WorkSpaceCenter1, x_val, -y_val, z_val), vmax, z100, tool2\WObj:=Wobj_1;
            
            ! Send "Done" confirmation (1 byte)
            SocketSend client_socket\Str:="D" \NoOfBytes:=1;
        ENDWHILE
    ENDPROC

    ! Use the appropriate connection procedure for your setup
    PROC SocketConnect()
        SocketCreate temp_socket;
        SocketBind temp_socket,"127.0.0.1",55000;
        SocketListen temp_socket;
        SocketAccept temp_socket,client_socket,\Time:=WAIT_MAX;
        TPWrite "Socket connection established.";
    ENDPROC
    
    ! For Real Robot:
    ! PROC SocketConnect()
    !     SocketCreate temp_socket;
    !     SocketBind temp_socket,"192.168.125.1",1025;
    !     SocketListen temp_socket;
    !     SocketAccept temp_socket,client_socket,\Time:=WAIT_MAX;
    !     TPWrite "Socket connection established.";
    ! ENDPROC

ENDMODULE
