//
//  Logger.swift
//  WenReader
//
//  Created by refactoring on 2025-12-10.
//

import Foundation

/// Simple logging utility for debugging and error tracking
enum Log {
    nonisolated static func info(_ msg: String) { 
        print("I:  \(msg)") 
    }
    
    nonisolated static func error(_ msg: String) { 
        print("E: \(msg)") 
    }
}
