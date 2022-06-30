// Copyright (c) 2019, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import 'package:flutter/material.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.lightGreen,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int weight = 0;
  int height = 0;
  double _bmi = 0;
  bool option = false;
  String result1 = "";
  String result2 = "";

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('비만도 계산기'),
      ),
      body: SingleChildScrollView(
      scrollDirection: Axis.vertical,
      child: Column(
        children: <Widget>[
          SizedBox(height: 10),
          TextField(
            decoration: InputDecoration(
                border: OutlineInputBorder(), labelText: '몸무게(kg)'),
            onChanged: (String w) {
              setState(() => weight = int.parse(w));
            },
          ),
          SizedBox(height: 10),
          TextField(
            decoration: InputDecoration(
                border: OutlineInputBorder(), labelText: '키(cm)'),
            onChanged: (String h) {
              setState(() => height = int.parse(h));
            },
          ),
          SizedBox(height: 10),
          TextButton(
              onPressed: () {
                setState(() {
                  option = true;
                  _bmi = weight / ((height / 100) * (height / 100));
                  result1 = 'BMI = ' + _bmi.toString();
                  result2 = checkbmi(_bmi);
                });
              },
              child: Text('계산하기')),
          SizedBox(height: 10),
          Text(result1, style: TextStyle(fontSize: 25)),
          SizedBox(height: 10),
          Text(result2, style: TextStyle(fontSize: 25)),
          _buildIcon(_bmi),
        ],
      ),
    ),
    );
  }

  String checkbmi(double bmi) {
    if (bmi < 20)
      return "저체중";
    else if (bmi < 25)
      return "정상";
    else if (bmi < 30)
      return "과체중";
    else
      return "비만";
  }

  Widget _buildIcon(double b) {
    if (option) {
      if (b >= 23) {
        return Icon(
          Icons.sentiment_very_dissatisfied,
          color: Colors.red,
          size: 100,
        );
      } else if (b >= 18.5) {
        return Icon(
          Icons.sentiment_satisfied,
          color: Colors.green,
          size: 100,
        );
      } else {
        return Icon(
          Icons.sentiment_dissatisfied,
          color: Colors.orange,
          size: 100,
        );
      }
    }
    else{
      return SizedBox(height:5);
    }
  }
}
